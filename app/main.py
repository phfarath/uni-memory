"""
Aethera Cortex v2.1 (Neon + Cloud Native MCP)
Plataforma de Memória Soberana & Gestão de Contexto.
Multi-Tenant | Auth Dinâmica | Postgres + pgvector | SSE Endpoint
"""

import os
import time
import threading
import uuid
import secrets
import requests
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Security, status, Query
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# --- CONFIGURAÇÃO ---
# A URL deve vir do .env: postgres://user:pass@endpoint.neon.tech/dbname?sslmode=require
DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Configuração de Modelo (Embedding Local)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aethera")

app = FastAPI(title="Aethera Cortex API v2.1 (Cloud Native)")

# --- SINGLETONS ---
logger.info(">>> [BOOT] Carregando Modelos Neurais...")
embed_model = SentenceTransformer(MODEL_NAME)
dim = embed_model.get_sentence_embedding_dimension() # 384 para MiniLM

# --- CONEXÃO BANCO ---
def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # Robustez: Remove prefixo 'psql ' se o usuário copiou o comando inteiro por engano
    clean_url = DATABASE_URL.strip()
    if clean_url.startswith("psql "):
        clean_url = clean_url.replace("psql ", "").strip()
        
    conn = psycopg2.connect(clean_url)
    return conn

# --- MODELOS PYDANTIC ---
class MemoryUpdate(BaseModel):
    content: str

class KeyRequest(BaseModel):
    owner_name: str
    tier: str = "free"

# --- BOOTSTRAP DO SCHEMA ---
def init_db():
    """Inicializa o Schema no Postgres (Neon)."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Habilita Extensão Vetorial
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # 2. Tabela de Memórias (Com vetor)
        c.execute(f'''CREATE TABLE IF NOT EXISTS memories (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp REAL,
                        embedding vector({dim})
                    )''')
        
        # 3. Tabela de Autenticação
        c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
                        key TEXT PRIMARY KEY,
                        owner_name TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at REAL,
                        tier TEXT DEFAULT 'free'
                    )''')
        
        conn.commit()

        # 4. Bootstrap Root Key
        c.execute("SELECT count(*) FROM api_keys")
        count = c.fetchone()[0]
        
        if count == 0:
            root_key = f"sk_aethera_root_{secrets.token_hex(16)}"
            logger.warning(f"\n{'='*60}")
            logger.warning(f" AETHERA CORTEX (NEON) - PRIMEIRA EXECUÇÃO")
            logger.warning(f" CHAVE MESTRA (ROOT): {root_key}")
            logger.warning(f" SALVE AGORA. NÃO APARECERÁ NOVAMENTE.")
            logger.warning(f"{'='*60}\n")
            
            c.execute("INSERT INTO api_keys (key, owner_name, is_active, created_at, tier) VALUES (%s, %s, TRUE, %s, 'root')", 
                      (root_key, "System Administrator", time.time()))
            conn.commit()
            
        c.close()
        conn.close()
        logger.info(">>> [BOOT] Postgres conectado e estruturado.")
        
    except Exception as e:
        logger.error(f"FATAL: Falha ao conectar no Neon: {e}")

# Call init on module load
init_db()

# --- SEGURANÇA (DUAL: HEADER OU QUERY) ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(
    api_key_header_val: str = Security(api_key_header),
    api_key_query_val: str = Query(None, alias="x-api-key") # Aceita ?x-api-key=...
):
    """
    Verifica a chave vinda do Header (API REST) ou da URL (MCP/SSE).
    """
    # Prioridade: Header > Query
    api_key = api_key_header_val or api_key_query_val
    
    if not api_key:
        raise HTTPException(status_code=403, detail="Acesso Negado: Chave ausente (Use header 'x-api-key' ou query param '?x-api-key=...')")
        
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT owner_name, tier FROM api_keys WHERE key = %s AND is_active = TRUE", (api_key,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {"key": api_key, "owner": row[0], "tier": row[1]}
    
    time.sleep(0.1) # Timing mitigation
    raise HTTPException(status_code=403, detail="Aethera Security: Chave inválida")

# --- MIDDLEWARE DE PROTEÇÃO DO MCP ---
class McpAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Se a rota começar com /mcp, exigimos autenticação
        if request.url.path.startswith("/mcp"):
            # Extrai a chave da Query String (Header é difícil em SSE)
            api_key = request.query_params.get("x-api-key")
            
            if not api_key:
                 # Tenta pegar do header como fallback
                api_key = request.headers.get("x-api-key")

            if not api_key:
                return JSONResponse(status_code=403, content={"detail": "MCP Auth Required: Passe ?x-api-key=..."})

            # Validação
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT 1 FROM api_keys WHERE key = %s AND is_active = TRUE", (api_key,))
                authorized = c.fetchone()
                conn.close()
                if not authorized:
                    return JSONResponse(status_code=403, content={"detail": "MCP Key Invalid"})
            except Exception as e:
                return JSONResponse(status_code=500, content={"detail": "Auth DB Error"})

        response = await call_next(request)
        return response

app.add_middleware(McpAuthMiddleware)

# --- CORE LÓGICO COMPARTILHADO ---

def add_memory_trace_logic(session_id: str, role: str, content: str):
    """Função síncrona/lógica pura para persistência."""
    try:
        # Vetorização
        vec = embed_model.encode([content])[0].tolist() 
        
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""INSERT INTO memories (session_id, role, content, timestamp, embedding) 
                     VALUES (%s, %s, %s, %s, %s)""",
                  (session_id, role, content, time.time(), vec))
        conn.commit()
        conn.close()
        logger.info(f"DEBUG [MEMORY] Trace persistido no Neon.")
    except Exception as e:
        logger.error(f"CRITICAL [MEMORY] Falha ao gravar no Postgres: {e}")
        raise e

def retrieve_context_logic(session_id: str, query: str, limit_k: int = 5) -> List[Dict]:
    context_items = []
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Curto Prazo
    c.execute("SELECT role, content FROM memories WHERE session_id = %s ORDER BY id DESC LIMIT 3", (session_id,))
    recent = c.fetchall()
    for r in reversed(recent):
        context_items.append({"source": "short_term", "role": r[0], "content": r[1]})
        
    # 2. Longo Prazo (pgvector)
    q_vec = embed_model.encode([query])[0].tolist()
    c.execute("""SELECT role, content, (embedding <=> %s::vector) as distance 
                 FROM memories 
                 ORDER BY distance ASC 
                 LIMIT %s""", 
              (q_vec, limit_k))
    
    vectors = c.fetchall()
    conn.close()
    
    for row in vectors:
        context_items.append({"source": "long_term", "role": row[0], "content": row[1]})
        
    return context_items

# Wrapper Async para FastAPI BackgroundTasks
def add_memory_trace(session_id: str, role: str, content: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(add_memory_trace_logic, session_id, role, content)

def execute_llm_call(model_name: str, system_context: str, user_query: str):
    if not OPENAI_API_KEY: return "Sem OPENAI_API_KEY configurada."
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name if model_name != "gpt-4" else "gpt-4o",
        "messages": [
            {"role": "system", "content": f"AETHERA CORTEX SYSTEM.\nCTX:\n{system_context}"},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.7
    }
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM Error: {e}"

# --- INTEGRAÇÃO MCP CLOUD (SSE - MANUAL) ---
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from starlette.responses import StreamingResponse

mcp_server = Server("Aethera Cloud")

# Definição das Ferramentas (Schema)
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="remember",
            description="Grava uma informação importante, fato, trecho de código ou preferência na memória de longo prazo.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "O conteúdo exato a ser lembrado."},
                    "category": {"type": "string", "description": "Tag para organização (ex: 'work', 'code'). Default: 'general'"}
                },
                "required": ["fact"]
            }
        ),
        Tool(
            name="recall",
            description="Busca na memória de longo prazo por informações relevantes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A pergunta ou tópico para pesquisar."}
                },
                "required": ["query"]
            }
        )
    ]

# Execução das Ferramentas
@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "remember":
        fact = arguments.get("fact")
        category = arguments.get("category", "general")
        try:
            enriched = f"[{category.upper()}] {fact}"
            # Chamada síncrona wrapper
            add_memory_trace_logic("mcp-sse-session", "user", enriched)
            return [TextContent(type="text", text="Memória salva com sucesso na Nuvem Aethera.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro interno: {e}")]
            
    elif name == "recall":
        query = arguments.get("query")
        try:
            items = retrieve_context_logic("mcp-sse-session", query)
            report = "MEMÓRIA RECUPERADA:\n"
            for item in items:
                report += f"- {item['content']}\n"
            return [TextContent(type="text", text=report)]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]
            
    raise ValueError(f"Ferramenta desconhecida: {name}")

# Armazenamento simples de sessões SSE (Em prod use Redis/Memcached)
sse_sessions: Dict[str, SseServerTransport] = {}

@app.get("/mcp/sse")
async def handle_sse(request: Request):
    # Cria um novo transporte para esta conexão
    transport = SseServerTransport("/mcp/messages")
    session_id = str(uuid.uuid4())
    
    # Armazena transporte
    sse_sessions[session_id] = transport
    
    async def event_generator():
        # 1. Envia URI de endpoint para o cliente
        yield f"event: endpoint\ndata: /mcp/messages?session={session_id}\n\n"
        
        # 2. Conecta e streama mensagens do servidor -> cliente
        try:
            async for message in transport.connect():
                yield f"event: message\ndata: {message.model_dump_json()}\n\n"
        except Exception as e:
            logger.error(f"SSE Connection Error: {e}")
        finally:
            # Limpeza
            if session_id in sse_sessions:
                del sse_sessions[session_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/mcp/messages")
async def handle_messages(request: Request):
    session_id = request.query_params.get("session")
    transport = sse_sessions.get(session_id)
    
    if not transport:
        return JSONResponse({"error": "Session not found"}, 404)
    
    # Processa mensagem JSON-RPC recebida
    try:
        body = await request.json()
        await transport.handle_post_message(mcp_server.create_initialization_options(), body)
        return JSONResponse({"status": "accepted"})
    except Exception as e:
        logger.error(f"Message Handling Error: {e}")
        return JSONResponse({"error": str(e)}, 500)

# --- ENDPOINTS REST CLASSICOS ---

@app.post("/admin/keys/create", tags=["Admin"])
async def create_api_key(new_key: KeyRequest, user: dict = Security(verify_api_key)):
    if user['tier'] != 'root': raise HTTPException(403, "Admin only")
    k = f"sk_aethera_{secrets.token_urlsafe(16)}"
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO api_keys (key, owner_name, created_at, tier) VALUES (%s, %s, %s, %s)",
              (k, new_key.owner_name, time.time(), new_key.tier))
    conn.commit()
    conn.close()
    return {"status": "created", "key": k}

@app.post("/admin/keys/revoke", tags=["Admin"])
async def revoke_api_key(target_key: str, user: dict = Security(verify_api_key)):
    if user['tier'] != 'root': raise HTTPException(403, "Admin only")
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE api_keys SET is_active = FALSE WHERE key = %s", (target_key,))
    conn.commit()
    conn.close()
    return {"status": "revoked"}

@app.post("/v1/chat/completions", tags=["Core"])
async def chat_protocol(request: Request, background_tasks: BackgroundTasks, user: dict = Security(verify_api_key)):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages: raise HTTPException(400, "Empty messages")
    
    user_query = messages[-1]['content']
    session_id = body.get("session_id", "default")
    model = body.get("model", "gpt-3.5-turbo")
    
    # 1. Retrieve
    ctx = retrieve_context_logic(session_id, user_query)
    
    if model == "memory-only":
        add_memory_trace(session_id, "user", user_query, background_tasks)
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "Memorized."}}]})
        
    # 2. Prompt
    sys_txt = "\n".join([f"-[{i['source']}]: {i['content']}" for i in ctx])
    
    # 3. Exec
    resp = execute_llm_call(model, sys_txt, user_query)
    
    # 4. Save
    add_memory_trace(session_id, "user", user_query, background_tasks)
    add_memory_trace(session_id, "assistant", resp, background_tasks)
    
    return JSONResponse({"choices": [{"message": {"role": "assistant", "content": resp}}]})

# CRUD
@app.get("/v1/memories")
async def list_memories(limit: int = 10, offset: int = 0, user: dict = Security(verify_api_key)):
    conn = get_db_connection()
    curr = conn.cursor(cursor_factory=RealDictCursor)
    curr.execute("SELECT id, role, content, timestamp, session_id FROM memories ORDER BY id DESC LIMIT %s OFFSET %s", (limit, offset))
    rows = curr.fetchall() # Returns list of dicts
    conn.close()
    return {"data": rows}

@app.delete("/v1/memories/{memory_id}")
async def delete_memory(memory_id: int, user: dict = Security(verify_api_key)):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    if deleted == 0: raise HTTPException(404, "Not found")
    return {"status": "deleted"}

@app.put("/v1/memories/{memory_id}")
async def update_memory(memory_id: int, update: MemoryUpdate, user: dict = Security(verify_api_key)):
    conn = get_db_connection()
    c = conn.cursor()
    new_vec = embed_model.encode([update.content])[0].tolist()
    c.execute("UPDATE memories SET content = %s, embedding = %s WHERE id = %s", 
              (update.content, new_vec, memory_id))
    updated = c.rowcount
    conn.commit()
    conn.close()
    if updated == 0: raise HTTPException(404, "Not found")
    return {"status": "updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

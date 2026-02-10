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
import contextvars
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
from app.rate_limiter import RateLimiter, ENDPOINT_ACTIONS, ACTION_REQUEST, cleanup_old_logs
from app.auto_capture import auto_capture_engine, process_pending_events, cleanup_old_auto_capture_events
from app.duplicate_prevention import check_duplicate, merge_memory

# --- CONFIGURAÇÃO ---
# A URL deve vir do .env: postgres://user:pass@endpoint.neon.tech/dbname?sslmode=require
DATABASE_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")  # 'test' or 'public'

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
    
    # Set schema search path (test vs public)
    if DB_SCHEMA and DB_SCHEMA != "public":
        c = conn.cursor()
        c.execute(f"SET search_path TO {DB_SCHEMA}, public")
        conn.commit()
        c.close()
    
    return conn

# --- MODELOS PYDANTIC ---
class MemoryUpdate(BaseModel):
    content: str

class KeyRequest(BaseModel):
    owner_name: str
    tier: str = "free"

class DuplicateCheckRequest(BaseModel):
    content: str
    session_id: str = "default"

class AutoCaptureToggle(BaseModel):
    session_id: str

class AutoCaptureEvent(BaseModel):
    session_id: str
    event_type: str
    event_data: dict

# --- BOOTSTRAP DO SCHEMA ---
def init_db():
    """Inicializa o Schema no Postgres (Neon)."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Habilita Extensão Vetorial
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # 2. Tabela de Memórias (Com vetor + Multi-tenancy)
        c.execute(f'''CREATE TABLE IF NOT EXISTS memories (
                        id SERIAL PRIMARY KEY,
                        owner_key TEXT NOT NULL,
                        workspace TEXT DEFAULT 'default',
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp REAL,
                        embedding vector({dim})
                    )''')
        
        # 2b. Indexes for multi-tenancy
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_key)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_workspace ON memories(owner_key, workspace)")

        # 2c. Vector similarity index for duplicate detection (HNSW)
        c.execute("CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops)")
        
        # 3. Tabela de Autenticação
        c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
                        key TEXT PRIMARY KEY,
                        owner_name TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at REAL,
                        tier TEXT DEFAULT 'free'
                    )''')
        
        # 4. Tabela de Definição de Tiers (Rate Limiting)
        c.execute('''CREATE TABLE IF NOT EXISTS tier_definitions (
                        tier TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        max_requests_per_day INTEGER DEFAULT 100,
                        max_memories INTEGER DEFAULT 1000,
                        max_embeddings_per_day INTEGER DEFAULT 50,
                        max_llm_calls_per_day INTEGER DEFAULT 0,
                        max_auto_capture_per_day INTEGER DEFAULT 1000,
                        priority INTEGER DEFAULT 0,
                        created_at REAL
                    )''')
        
        # 5. Tabela de Logs de Uso (Usage Tracking)
        c.execute('''CREATE TABLE IF NOT EXISTS usage_logs (
                        id SERIAL PRIMARY KEY,
                        api_key TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        response_status INTEGER,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )''')
        
        # Indexes for usage_logs
        c.execute("CREATE INDEX IF NOT EXISTS idx_usage_key_time ON usage_logs(api_key, timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_usage_action ON usage_logs(action_type, timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_usage_cleanup ON usage_logs(created_at)")

        # 6. Tabela de Eventos de Auto-Capture
        c.execute('''CREATE TABLE IF NOT EXISTS auto_capture_events (
                        id SERIAL PRIMARY KEY,
                        owner_key TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        event_data JSONB NOT NULL,
                        captured_at TIMESTAMPTZ DEFAULT NOW(),
                        processed BOOLEAN DEFAULT FALSE,
                        memory_id INTEGER REFERENCES memories(id)
                    )''')

        # Indexes for auto_capture_events
        c.execute("CREATE INDEX IF NOT EXISTS idx_auto_capture_session ON auto_capture_events(session_id, captured_at DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_auto_capture_processed ON auto_capture_events(processed, captured_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_auto_capture_owner ON auto_capture_events(owner_key)")

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
        
        # 5. Bootstrap Default Tiers (if empty)
        c.execute("SELECT count(*) FROM tier_definitions")
        tier_count = c.fetchone()[0]
        
        if tier_count == 0:
            now = time.time()
            # (tier, display_name, max_requests, max_memories, max_embeddings, max_llm_calls, max_auto_capture, priority, created_at)
            default_tiers = [
                ('free', 'Free', 100, 1000, 50, 0, 100, 0, now),
                ('pro', 'Pro', 5000, 50000, 1000, 100, 5000, 1, now),
                ('team', 'Team', 50000, 500000, 10000, 1000, 50000, 2, now),
                ('root', 'Admin', -1, -1, -1, -1, -1, 99, now),  # -1 = unlimited
            ]
            for tier_data in default_tiers:
                c.execute("""INSERT INTO tier_definitions
                            (tier, display_name, max_requests_per_day, max_memories,
                             max_embeddings_per_day, max_llm_calls_per_day, max_auto_capture_per_day, priority, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""", tier_data)
            conn.commit()
            logger.info(">>> [BOOT] Tiers padrão criados (free, pro, team, root).")
            
        c.close()
        conn.close()
        logger.info(">>> [BOOT] Postgres conectado e estruturado.")
        
    except Exception as e:
        logger.error(f"FATAL: Falha ao conectar no Neon: {e}")

# Call init on module load
init_db()

# --- RATE LIMITER SINGLETON ---
rate_limiter = RateLimiter(get_db_connection)
logger.info(">>> [BOOT] Rate Limiter inicializado.")

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
        # Protegemos APENAS a porta de entrada (Handshake SSE)
        # As mensagens subsequentes (/messages) são validadas pelo Session ID do FastMCP
        if request.url.path.endswith("/sse"):
            # 1. Pega a chave
            api_key = request.query_params.get("x-api-key") or request.headers.get("x-api-key")
            
            if not api_key:
                return JSONResponse(status_code=403, content={"detail": "Missing Key in SSE Handshake"})

            # 2. Validação (Fail-Safe)
            try:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT 1 FROM api_keys WHERE key = %s AND is_active = TRUE", (api_key,))
                authorized = c.fetchone()
                conn.close()
                
                if not authorized:
                    return JSONResponse(status_code=403, content={"detail": "Invalid Key"})
                    
            except Exception as e:
                logger.error(f"AUTH ERROR: {e}")
                return JSONResponse(status_code=500, content={"detail": "Auth System Error"})

        return await call_next(request)

# --- RATE LIMIT MIDDLEWARE ---
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware que verifica rate limits em endpoints protegidos."""
    
    # Endpoints que NÃO são rate-limited
    SKIP_PATHS = {"/docs", "/openapi.json", "/health", "/", "/redoc"}
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        
        # Skip non-protected endpoints
        if path in self.SKIP_PATHS or path.endswith("/docs"):
            return await call_next(request)
        
        # Get API key (from header or query)
        api_key = request.headers.get("x-api-key") or request.query_params.get("x-api-key")
        
        if not api_key:
            # Let the auth middleware handle missing keys
            return await call_next(request)
        
        # Get user tier
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("SELECT tier FROM api_keys WHERE key = %s AND is_active = TRUE", (api_key,))
            row = c.fetchone()
            conn.close()
            
            if not row:
                return await call_next(request)  # Let auth middleware reject
            
            tier = row[0]
        except Exception:
            return await call_next(request)
        
        # Determine action type from endpoint
        endpoint_key = f"{method} {path}"
        action_type, _ = ENDPOINT_ACTIONS.get(endpoint_key, (ACTION_REQUEST, 1))
        
        # Check rate limit
        is_allowed, usage_info = rate_limiter.check_limit(api_key, action_type, tier)
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "tier": tier,
                    "usage": usage_info,
                    "message": f"Limite de {usage_info['limit']} {action_type}/dia atingido. Upgrade para aumentar."
                },
                headers={"Retry-After": "86400"}  # 24 hours
            )
        
        # Execute request
        response = await call_next(request)
        
        # Log usage (fire-and-forget)
        rate_limiter.log_usage(
            api_key=api_key,
            endpoint=path,
            method=method,
            action_type=action_type,
            status=response.status_code
        )
        
        return response

# --- MIDDLEWARES ---
from fastapi.middleware.cors import CORSMiddleware

# CORS (Permitir tudo para desenvolvimento local/teste)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(McpAuthMiddleware)

# --- CORE LÓGICO COMPARTILHADO ---

def add_memory_trace_logic(owner_key: str, session_id: str, role: str, content: str, workspace: str = "default", force: bool = False):
    """Função síncrona/lógica pura para persistência. Inclui owner_key para multi-tenancy e prevenção de duplicatas."""
    try:
        # Vetorização (calculado uma vez, reutilizado para check + insert)
        vec = embed_model.encode([content])[0].tolist()

        # Duplicate check (skip if force=True)
        if not force:
            dup_result = check_duplicate(vec, owner_key, get_db_connection)
            if dup_result.is_duplicate:
                merge_memory(dup_result.existing_id, owner_key, get_db_connection)
                logger.info(
                    f"DEBUG [MEMORY] Duplicata detectada (similarity={dup_result.similarity:.4f}). "
                    f"Merge realizado com memory_id={dup_result.existing_id}."
                )
                return {"action": "merged", "existing_id": dup_result.existing_id,
                        "similarity": dup_result.similarity}

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("""INSERT INTO memories (owner_key, workspace, session_id, role, content, timestamp, embedding)
                     VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                  (owner_key, workspace, session_id, role, content, time.time(), vec))
        conn.commit()
        conn.close()
        logger.info(f"DEBUG [MEMORY] Trace persistido para owner {owner_key[:20]}...")
        return {"action": "created"}
    except Exception as e:
        logger.error(f"CRITICAL [MEMORY] Falha ao gravar no Postgres: {e}")
        raise e

def retrieve_context_logic(owner_key: str, session_id: str, query: str, workspace: str = None, limit_k: int = 5) -> List[Dict]:
    """Recupera contexto filtrado pelo owner_key e opcionalmente por workspace."""
    context_items = []
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Curto Prazo - filter by owner
    c.execute("""SELECT role, content FROM memories 
                 WHERE owner_key = %s AND session_id = %s 
                 ORDER BY id DESC LIMIT 3""", (owner_key, session_id))
    recent = c.fetchall()
    for r in reversed(recent):
        context_items.append({"source": "short_term", "role": r[0], "content": r[1]})
        
    # 2. Longo Prazo (pgvector) - filter by owner and optionally workspace
    q_vec = embed_model.encode([query])[0].tolist()
    
    if workspace:
        c.execute("""SELECT role, content, (embedding <=> %s::vector) as distance 
                     FROM memories 
                     WHERE owner_key = %s AND workspace = %s
                     ORDER BY distance ASC 
                     LIMIT %s""", 
                  (q_vec, owner_key, workspace, limit_k))
    else:
        c.execute("""SELECT role, content, (embedding <=> %s::vector) as distance 
                     FROM memories 
                     WHERE owner_key = %s
                     ORDER BY distance ASC 
                     LIMIT %s""", 
                  (q_vec, owner_key, limit_k))
    
    vectors = c.fetchall()
    conn.close()
    
    for row in vectors:
        context_items.append({"source": "long_term", "role": row[0], "content": row[1]})
        
    return context_items

# Wrapper Async para FastAPI BackgroundTasks
def add_memory_trace(owner_key: str, session_id: str, role: str, content: str, background_tasks: BackgroundTasks, workspace: str = "default"):
    background_tasks.add_task(add_memory_trace_logic, owner_key, session_id, role, content, workspace)


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

# --- INTEGRAÇÃO MCP CLOUD (MANUAL - ANYIO STREAMS) ---
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from starlette.responses import StreamingResponse
import asyncio
import anyio

mcp_server = Server("Aethera Cloud")

# Definição das Ferramentas (Schema)
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="remember",
            description="Grava uma informação importante na memória de longo prazo. Detecta duplicatas automaticamente.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "O conteúdo exato a ser lembrado."},
                    "category": {"type": "string", "description": "Tag para organização (ex: 'work', 'code'). Default: 'general'"},
                    "force": {"type": "boolean", "description": "Se true, grava mesmo que duplicata exista. Default: false"}
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
        ),
        Tool(
            name="list_recent",
            description="Lista as memórias mais recentes armazenadas.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Número máximo de memórias para retornar. Default: 10"}
                },
                "required": []
            }
        ),
        Tool(
            name="update_memory",
            description="Atualiza o conteúdo de uma memória existente pelo seu ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "integer", "description": "O ID da memória a ser atualizada."},
                    "new_content": {"type": "string", "description": "O novo conteúdo para substituir o antigo."}
                },
                "required": ["memory_id", "new_content"]
            }
        ),
        Tool(
            name="forget",
            description="Remove uma memória específica pelo seu ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "integer", "description": "O ID da memória a ser removida."}
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="enable_auto_capture",
            description="Ativa captura automática de contexto para a sessão atual. Quando ativado, comandos, edições de arquivos, erros e decisões são automaticamente gravados como memórias.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "ID da sessão para ativar auto-capture."}
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="disable_auto_capture",
            description="Desativa captura automática de contexto para a sessão.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "ID da sessão para desativar auto-capture."}
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="auto_capture_status",
            description="Verifica o status de auto-capture para uma sessão.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "ID da sessão para verificar."}
                },
                "required": ["session_id"]
            }
        )
    ]

# Execução das Ferramentas
@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Get owner_key from context (set by POST handler)
    owner_key = current_api_key.get()
    if not owner_key:
        return [TextContent(type="text", text="Erro: Sessão não autenticada.")]
    
    if name == "remember":
        fact = arguments.get("fact")
        category = arguments.get("category", "general")
        workspace = arguments.get("workspace", "default")
        force = arguments.get("force", False)
        try:
            enriched = f"[{category.upper()}] {fact}"
            result = add_memory_trace_logic(owner_key, "mcp-session", "user", enriched, workspace, force=force)

            if result and result.get("action") == "merged":
                return [TextContent(
                    type="text",
                    text=f"Memória similar já existe (similaridade: {result['similarity']:.1%}). "
                         f"Timestamp atualizado na memória existente (ID: {result['existing_id']}). "
                         f"Use force=true para gravar mesmo assim."
                )]

            return [TextContent(type="text", text="Memória salva com sucesso na Nuvem Aethera.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro interno: {e}")]
            
    elif name == "recall":
        query = arguments.get("query")
        workspace = arguments.get("workspace")  # None = search all workspaces
        try:
            items = retrieve_context_logic(owner_key, "mcp-session", query, workspace)
            report = "MEMÓRIA RECUPERADA:\n"
            for item in items:
                report += f"- {item['content']}\n"
            return [TextContent(type="text", text=report)]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]
    
    elif name == "list_recent":
        limit = arguments.get("limit", 10)
        workspace = arguments.get("workspace")
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Filter by owner (and optionally workspace)
            if workspace:
                c.execute("""SELECT id, role, content, timestamp, workspace FROM memories 
                             WHERE owner_key = %s AND workspace = %s 
                             ORDER BY id DESC LIMIT %s""", (owner_key, workspace, limit))
            else:
                c.execute("""SELECT id, role, content, timestamp, workspace FROM memories 
                             WHERE owner_key = %s 
                             ORDER BY id DESC LIMIT %s""", (owner_key, limit))
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                return [TextContent(type="text", text="Nenhuma memória encontrada.")]
            
            report = f"ÚLTIMAS {len(rows)} MEMÓRIAS:\n"
            for row in rows:
                mem_id, role, content, ts, ws = row
                # Truncate long content for display
                content_preview = content[:100] + "..." if len(content) > 100 else content
                report += f"[ID:{mem_id}] ({ws}) {content_preview}\n"
            return [TextContent(type="text", text=report)]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]
    
    elif name == "update_memory":
        memory_id = arguments.get("memory_id")
        new_content = arguments.get("new_content")
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Verify ownership before update
            c.execute("SELECT id FROM memories WHERE id = %s AND owner_key = %s", (memory_id, owner_key))
            if not c.fetchone():
                conn.close()
                return [TextContent(type="text", text=f"Memória {memory_id} não encontrada ou sem permissão.")]
            
            # Generate new embedding for the updated content
            new_vec = embed_model.encode([new_content])[0].tolist()
            c.execute("UPDATE memories SET content = %s, embedding = %s WHERE id = %s AND owner_key = %s", 
                      (new_content, new_vec, memory_id, owner_key))
            updated = c.rowcount
            conn.commit()
            conn.close()
            
            if updated == 0:
                return [TextContent(type="text", text=f"Memória com ID {memory_id} não encontrada.")]
            return [TextContent(type="text", text=f"Memória {memory_id} atualizada com sucesso.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]
    
    elif name == "forget":
        memory_id = arguments.get("memory_id")
        try:
            conn = get_db_connection()
            c = conn.cursor()

            # Verify ownership before delete
            c.execute("DELETE FROM memories WHERE id = %s AND owner_key = %s", (memory_id, owner_key))
            deleted = c.rowcount
            conn.commit()
            conn.close()

            if deleted == 0:
                return [TextContent(type="text", text=f"Memória {memory_id} não encontrada ou sem permissão.")]
            return [TextContent(type="text", text=f"Memória {memory_id} removida com sucesso.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]

    elif name == "enable_auto_capture":
        session_id = arguments.get("session_id")
        try:
            auto_capture_engine.enable_for_session(session_id, owner_key)
            return [TextContent(type="text", text=f"Auto-capture ativado para sessão '{session_id}'. Comandos, edições, erros e decisões serão capturados automaticamente.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]

    elif name == "disable_auto_capture":
        session_id = arguments.get("session_id")
        try:
            auto_capture_engine.disable_for_session(session_id)
            return [TextContent(type="text", text=f"Auto-capture desativado para sessão '{session_id}'.")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]

    elif name == "auto_capture_status":
        session_id = arguments.get("session_id")
        try:
            is_enabled = auto_capture_engine.is_enabled(session_id)
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("""
                SELECT COUNT(*) as total,
                       COUNT(CASE WHEN processed THEN 1 END) as processed
                FROM auto_capture_events
                WHERE session_id = %s AND owner_key = %s
            """, (session_id, owner_key))
            stats = c.fetchone()
            conn.close()

            status_text = f"Status Auto-Capture para sessão '{session_id}':\n"
            status_text += f"- Ativo: {'Sim' if is_enabled else 'Não'}\n"
            status_text += f"- Total de eventos: {stats[0]}\n"
            status_text += f"- Eventos processados: {stats[1]}"
            return [TextContent(type="text", text=status_text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]

    raise ValueError(f"Ferramenta desconhecida: {name}")

# --- STREAMABLE HTTP TRANSPORT (MCP 2025) ---
# Stores for managing sessions
streamable_sessions: Dict[str, dict] = {}

@app.post("/mcp")
async def handle_streamable_http(request: Request):
    """
    Streamable HTTP transport endpoint (MCP 2025-03-26 spec).
    Handles JSON-RPC messages via POST with optional SSE streaming response.
    """
    # Validate API key
    api_key = request.query_params.get("x-api-key") or request.headers.get("x-api-key")
    if not api_key:
        return JSONResponse({"error": "Missing API key"}, status_code=403)
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT 1 FROM api_keys WHERE key = %s AND is_active = TRUE", (api_key,))
        authorized = c.fetchone()
        conn.close()
        if not authorized:
            return JSONResponse({"error": "Invalid API key"}, status_code=403)
    except Exception as e:
        logger.error(f"Auth error: {e}")
        return JSONResponse({"error": "Auth system error"}, status_code=500)
    
    # Get session ID from header or create new one
    session_id = request.headers.get("mcp-session-id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logger.info(f"DEBUG: Streamable HTTP request for session {session_id}")
    
    try:
        body = await request.json()
        logger.info(f"DEBUG: Received JSON-RPC: {str(body)[:200]}...")
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}, status_code=400)
    
    # Check Accept header to determine response type
    accept_header = request.headers.get("accept", "application/json")
    wants_sse = "text/event-stream" in accept_header
    
    # Handle the JSON-RPC message
    from mcp.types import JSONRPCMessage, JSONRPCRequest, JSONRPCNotification
    from pydantic import TypeAdapter
    
    # Log raw body for debugging
    logger.info(f"DEBUG: Raw body type check - id={body.get('id')}, method={body.get('method')}")
    
    # Process directly from body instead of parsing through JSONRPCMessage union
    # This is more reliable than dealing with the union type
    method = body.get("method")
    request_id = body.get("id")
    params = body.get("params", {}) or {}
    
    # Notification = has method but no id
    # Request = has method AND id
    # Response = has no method (has result or error)
    
    is_request = method is not None and request_id is not None
    is_notification = method is not None and request_id is None
    
    logger.info(f"DEBUG: is_request={is_request}, is_notification={is_notification}, method={method}, id={request_id}")
    
    if is_request:
        # Process the request and return response
        logger.info(f"DEBUG: Processing method '{method}' with id {request_id}")
        
        try:
            if method == "initialize":
                # Handle initialize request
                response_result = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {"listChanged": False}
                    },
                    "serverInfo": {
                        "name": "Aethera Cloud",
                        "version": "2.1.0"
                    }
                }
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": response_result
                }
                # Include session ID in response header
                headers = {"mcp-session-id": session_id}
                return JSONResponse(response, headers=headers)
                
            elif method == "tools/list":
                # Return list of available tools
                tools = await list_tools()
                tools_json = [{"name": t.name, "description": t.description, "inputSchema": t.inputSchema} for t in tools]
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": tools_json}
                }
                return JSONResponse(response)
                
            elif method == "tools/call":
                # Execute a tool
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                
                logger.info(f"DEBUG: Calling tool '{tool_name}' with args {tool_args}")
                
                result = await call_tool(tool_name, tool_args)
                content_json = [{"type": c.type, "text": c.text} for c in result]
                
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": content_json, "isError": False}
                }
                return JSONResponse(response)
                
            elif method == "ping":
                response = {"jsonrpc": "2.0", "id": request_id, "result": {}}
                return JSONResponse(response)
                
            else:
                # Method not found
                response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
                return JSONResponse(response, status_code=200)
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
            return JSONResponse(response, status_code=200)
    
    # Handle notifications (method but no id)
    if is_notification:
        logger.info(f"DEBUG: Received notification '{method}'")
        # For notifications/initialized, just accept
        return JSONResponse(status_code=202, content=None)
    
    # For responses or unknown, just accept
    return JSONResponse(status_code=202, content=None)

# Also support GET for backwards compatibility (returns SSE stream with endpoint info)
@app.get("/mcp")
async def handle_mcp_get(request: Request):
    """
    Backwards compatibility: GET requests fall back to SSE transport.
    Redirects to /mcp/sse endpoint behavior.
    """
    # Forward to the SSE handler
    return await handle_sse(request)

# --- OLD SSE TRANSPORT (FOR BACKWARDS COMPATIBILITY) ---
# Estrutura para segurar as pontas dos streams
class SseSession:
    def __init__(self, post_sink, sse_source, api_key: str = None):
        self.post_sink = post_sink   # Onde escrevemos o que vem do POST
        self.sse_source = sse_source # De onde lemos para mandar pro SSE
        self.api_key = api_key       # Owner key for multi-tenancy

# Store sessions
sse_sessions: Dict[str, SseSession] = {}

# Context variable to pass API key to tool handlers during MCP request processing
current_api_key: contextvars.ContextVar[str] = contextvars.ContextVar('current_api_key', default=None)

@app.get("/mcp/sse")
async def handle_sse(request: Request):
    session_id = str(uuid.uuid4())
    api_key = request.query_params.get("x-api-key") or request.headers.get("x-api-key")
    logger.info(f"DEBUG: Starting SSE Session {session_id} for key {api_key[:20] if api_key else 'NONE'}...")

    # Cria canais AnyIO
    # Docs: send_stream, receive_stream = create_memory_object_stream(buffer_size)
    
    # Canal 1: Cliente POST (Send) -> Server (Recv)
    post_send, server_recv = anyio.create_memory_object_stream(10)
    
    # Canal 2: Server (Send) -> Cliente SSE (Recv)
    server_send, sse_recv = anyio.create_memory_object_stream(10)
    
    # Armazena as pontas 'Client-Side' na sessão (incluindo API key para multi-tenancy)
    # post_sink = onde o endpoint POST escreve (send stream)
    # sse_source = onde o endpoint GET lê (recv stream)
    sse_sessions[session_id] = SseSession(post_send, sse_recv, api_key)
    async def run_server_loop():
        logger.info(f"DEBUG: MCP Server Loop STARTED for {session_id}")
        try:
            # Server.run() espera (read_stream, write_stream, options)
            options = mcp_server.create_initialization_options()
            await mcp_server.run(server_recv, server_send, options)
            logger.info(f"DEBUG: MCP Server Loop ENDED cleanly for {session_id}")
        except Exception as e:
            logger.error(f"MCP Server Loop Error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.create_task(run_server_loop())

    async def event_generator():
        # Constrói a URL ABSOLUTA do endpoint
        # Forçamos 127.0.0.1 para evitar problemas de DNS (localhost vs ::1)
        base_url = os.environ.get("MCP_PUBLIC_URL", "http://127.0.0.1:8001")
        endpoint_url = f"{base_url}/mcp/messages?session={session_id}"
        
        # Preserva a chave de API
        if api_key:
            endpoint_url += f"&x-api-key={api_key}"
            
        logger.info(f"DEBUG: Yielding endpoint: {endpoint_url}")
        yield f"event: endpoint\ndata: {endpoint_url}\n\n"
        
        try:
            async with sse_sessions[session_id].sse_source:
                async for message in sse_sessions[session_id].sse_source:
                    # Message é um objeto JSONRPCMessage, precisamos serializar
                    # A lib usually returns objects.
                    # Pydantic v2 dump
                    if hasattr(message, 'model_dump_json'):
                        data = message.model_dump_json()
                    else:
                        data = str(message)
                        
                    logger.info(f"DEBUG: Sending SSE Message to client: {data[:100]}...")
                    yield f"event: message\ndata: {data}\n\n"
        except anyio.EndOfStream:
             logger.info("Stream ended")
        except asyncio.CancelledError:
            logger.info("Client disconnected")
        except Exception as e:
             logger.error(f"Event Generator Error: {e}")
        finally:
            logger.info(f"Cleaning up session {session_id}")
            if session_id in sse_sessions:
                del sse_sessions[session_id]

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/mcp/messages")
async def handle_messages(request: Request):
    session_id = request.query_params.get("session")
    session = sse_sessions.get(session_id)
    
    logger.info(f"DEBUG: POST /messages received for session {session_id}")
    
    if not session:
        logger.warning(f"DEBUG: Session {session_id} not found in {list(sse_sessions.keys())}")
        return JSONResponse({"error": "Session not found"}, 404)
    
    # Set API key context for tool handlers
    token = current_api_key.set(session.api_key)
    
    try:
        body = await request.json()
        
        # Validar e Parsear usando a lib MCP
        from mcp.types import JSONRPCMessage
        from pydantic import TypeAdapter
        adapter = TypeAdapter(JSONRPCMessage)
        msg_obj = adapter.validate_python(body)
        
        # Escreve no sink (post_sink)
        # É um MemoryObjectSendStream
        logger.info(f"DEBUG: Sending to server input sink...")
        await session.post_sink.send(msg_obj)
        logger.info(f"DEBUG: Sent to server input sink.")
        
        return JSONResponse({"status": "accepted"})
    except Exception as e:
        logger.error(f"Message Post Error: {e}")
        return JSONResponse({"error": str(e)}, 500)
    finally:
        # Reset context
        current_api_key.reset(token)

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
    workspace = body.get("workspace", "default")
    owner_key = user["key"]
    
    # 1. Retrieve (filtered by owner)
    ctx = retrieve_context_logic(owner_key, session_id, user_query, workspace)
    
    if model == "memory-only":
        add_memory_trace(owner_key, session_id, "user", user_query, background_tasks, workspace)
        return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "Memorized."}}]})
        
    # 2. Prompt
    sys_txt = "\n".join([f"-[{i['source']}]: {i['content']}" for i in ctx])
    
    # 3. Exec
    resp = execute_llm_call(model, sys_txt, user_query)
    
    # 4. Save (with owner_key)
    add_memory_trace(owner_key, session_id, "user", user_query, background_tasks, workspace)
    add_memory_trace(owner_key, session_id, "assistant", resp, background_tasks, workspace)
    
    return JSONResponse({"choices": [{"message": {"role": "assistant", "content": resp}}]})

# CRUD (filtered by owner)
@app.get("/v1/memories")
async def list_memories(limit: int = 10, offset: int = 0, workspace: str = None, user: dict = Security(verify_api_key)):
    owner_key = user["key"]
    conn = get_db_connection()
    curr = conn.cursor(cursor_factory=RealDictCursor)
    
    if workspace:
        curr.execute("""SELECT id, role, content, timestamp, session_id, workspace FROM memories 
                        WHERE owner_key = %s AND workspace = %s 
                        ORDER BY id DESC LIMIT %s OFFSET %s""", (owner_key, workspace, limit, offset))
    else:
        curr.execute("""SELECT id, role, content, timestamp, session_id, workspace FROM memories 
                        WHERE owner_key = %s 
                        ORDER BY id DESC LIMIT %s OFFSET %s""", (owner_key, limit, offset))
    rows = curr.fetchall()
    conn.close()
    return {"data": rows}

@app.delete("/v1/memories/{memory_id}")
async def delete_memory(memory_id: int, user: dict = Security(verify_api_key)):
    owner_key = user["key"]
    conn = get_db_connection()
    c = conn.cursor()
    # Verify ownership before delete
    c.execute("DELETE FROM memories WHERE id = %s AND owner_key = %s", (memory_id, owner_key))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    if deleted == 0: raise HTTPException(404, "Not found or no permission")
    return {"status": "deleted"}

@app.put("/v1/memories/{memory_id}")
async def update_memory(memory_id: int, update: MemoryUpdate, user: dict = Security(verify_api_key)):
    owner_key = user["key"]
    conn = get_db_connection()
    c = conn.cursor()
    
    # Verify ownership before update
    c.execute("SELECT id FROM memories WHERE id = %s AND owner_key = %s", (memory_id, owner_key))
    if not c.fetchone():
        conn.close()
        raise HTTPException(404, "Not found or no permission")
    
    new_vec = embed_model.encode([update.content])[0].tolist()
    c.execute("UPDATE memories SET content = %s, embedding = %s WHERE id = %s AND owner_key = %s", 
              (update.content, new_vec, memory_id, owner_key))
    updated = c.rowcount
    conn.commit()
    conn.close()
    if updated == 0: raise HTTPException(404, "Not found")
    return {"status": "updated"}

@app.post("/v1/memories/check-duplicate", tags=["Core"])
async def check_memory_duplicate(req: DuplicateCheckRequest, user: dict = Security(verify_api_key)):
    """Verifica se uma memória similar já existe antes de gravar."""
    owner_key = user["key"]
    vec = embed_model.encode([req.content])[0].tolist()
    result = check_duplicate(vec, owner_key, get_db_connection)
    return {
        "is_duplicate": result.is_duplicate,
        "existing_id": result.existing_id,
        "existing_content": result.existing_content,
        "similarity": round(result.similarity, 4) if result.similarity > 0 else 0
    }

# --- USAGE & TIER MANAGEMENT ENDPOINTS ---

class TierUpgradeRequest(BaseModel):
    new_tier: str

@app.get("/v1/usage", tags=["Usage"])
async def get_usage(user: dict = Security(verify_api_key)):
    """Returns current usage stats for the authenticated user."""
    return rate_limiter.get_full_usage_stats(user["key"], user["tier"])

@app.get("/admin/usage/stats", tags=["Admin"])
async def get_admin_usage_stats(user: dict = Security(verify_api_key)):
    """Admin-only: Get aggregated usage stats across all users."""
    if user['tier'] != 'root':
        raise HTTPException(403, "Admin only")
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # Total users by tier
    c.execute("""
        SELECT tier, COUNT(*) as count 
        FROM api_keys 
        WHERE is_active = TRUE 
        GROUP BY tier
    """)
    tier_counts = {row[0]: row[1] for row in c.fetchall()}
    
    # Active today (users with usage logs today)
    c.execute("""
        SELECT COUNT(DISTINCT api_key) 
        FROM usage_logs 
        WHERE DATE(created_at) = CURRENT_DATE
    """)
    active_today = c.fetchone()[0]
    
    # Requests by tier today
    c.execute("""
        SELECT ak.tier, COUNT(ul.id) as requests
        FROM usage_logs ul
        JOIN api_keys ak ON ul.api_key = ak.key
        WHERE DATE(ul.created_at) = CURRENT_DATE
        GROUP BY ak.tier
    """)
    requests_by_tier = {row[0]: row[1] for row in c.fetchall()}
    
    conn.close()
    
    return {
        "total_users": sum(tier_counts.values()),
        "active_today": active_today,
        "by_tier": {
            tier: {
                "users": tier_counts.get(tier, 0),
                "requests_today": requests_by_tier.get(tier, 0)
            }
            for tier in ["free", "pro", "team", "root"]
        }
    }

@app.post("/admin/users/{api_key}/upgrade", tags=["Admin"])
async def upgrade_user_tier(api_key: str, request: TierUpgradeRequest, user: dict = Security(verify_api_key)):
    """Admin-only: Upgrade or downgrade a user's tier."""
    if user['tier'] != 'root':
        raise HTTPException(403, "Admin only")
    
    # Validate new tier exists
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT tier FROM tier_definitions WHERE tier = %s", (request.new_tier,))
    if not c.fetchone():
        conn.close()
        raise HTTPException(400, f"Invalid tier: {request.new_tier}")
    
    # Get old tier
    c.execute("SELECT tier FROM api_keys WHERE key = %s", (api_key,))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "API key not found")
    
    old_tier = row[0]
    
    # Update tier
    c.execute("UPDATE api_keys SET tier = %s WHERE key = %s", (request.new_tier, api_key))
    conn.commit()
    conn.close()
    
    # Refresh rate limiter cache
    rate_limiter._refresh_tier_cache()
    
    logger.info(f"[ADMIN] Tier upgrade: {api_key[:20]}... | {old_tier} -> {request.new_tier}")
    
    return {
        "status": "upgraded",
        "key": api_key[:20] + "...",
        "old_tier": old_tier,
        "new_tier": request.new_tier
    }

@app.get("/admin/tiers", tags=["Admin"])
async def list_tiers(user: dict = Security(verify_api_key)):
    """List all available tiers and their limits."""
    if user['tier'] != 'root':
        raise HTTPException(403, "Admin only")

    conn = get_db_connection()
    c = conn.cursor(cursor_factory=RealDictCursor)
    c.execute("SELECT * FROM tier_definitions ORDER BY priority")
    rows = c.fetchall()
    conn.close()

    return {"tiers": rows}

# --- AUTO-CAPTURE ENDPOINTS ---

@app.post("/v1/auto-capture/enable", tags=["Auto-Capture"])
async def enable_auto_capture(toggle: AutoCaptureToggle, user: dict = Security(verify_api_key)):
    """Enable auto-capture for a session."""
    owner_key = user["key"]
    auto_capture_engine.enable_for_session(toggle.session_id, owner_key)
    return {"status": "enabled", "session_id": toggle.session_id}

@app.post("/v1/auto-capture/disable", tags=["Auto-Capture"])
async def disable_auto_capture(toggle: AutoCaptureToggle, user: dict = Security(verify_api_key)):
    """Disable auto-capture for a session."""
    auto_capture_engine.disable_for_session(toggle.session_id)
    return {"status": "disabled", "session_id": toggle.session_id}

@app.get("/v1/auto-capture/status", tags=["Auto-Capture"])
async def get_auto_capture_status(session_id: str, user: dict = Security(verify_api_key)):
    """Get auto-capture status and stats for a session."""
    owner_key = user["key"]
    is_enabled = auto_capture_engine.is_enabled(session_id)

    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*) as total,
               COUNT(CASE WHEN processed THEN 1 END) as processed
        FROM auto_capture_events
        WHERE session_id = %s AND owner_key = %s
    """, (session_id, owner_key))
    stats = c.fetchone()
    conn.close()

    return {
        "session_id": session_id,
        "enabled": is_enabled,
        "total_events": stats[0],
        "processed_events": stats[1],
        "pending_events": stats[0] - stats[1]
    }

@app.post("/v1/auto-capture/event", tags=["Auto-Capture"])
async def capture_event(event: AutoCaptureEvent, user: dict = Security(verify_api_key)):
    """Capture an event for processing. Only works if auto-capture is enabled for the session."""
    owner_key = user["key"]

    # Check if auto-capture is enabled for this session
    if not auto_capture_engine.is_enabled(event.session_id):
        return {"status": "skipped", "reason": "auto-capture not enabled for this session"}

    # Check if owner matches
    session_owner = auto_capture_engine.get_owner_key(event.session_id)
    if session_owner != owner_key:
        raise HTTPException(403, "Session belongs to different owner")

    # Insert event into database for background processing
    try:
        conn = get_db_connection()
        c = conn.cursor()
        import json
        c.execute("""
            INSERT INTO auto_capture_events (owner_key, session_id, event_type, event_data)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (owner_key, event.session_id, event.event_type, json.dumps(event.event_data)))
        event_id = c.fetchone()[0]
        conn.commit()
        conn.close()

        return {"status": "captured", "event_id": event_id, "session_id": event.session_id}
    except Exception as e:
        logger.error(f"[AUTO_CAPTURE] Failed to capture event: {e}")
        raise HTTPException(500, f"Failed to capture event: {e}")

# --- BACKGROUND WORKER SETUP ---
import schedule

def run_auto_capture_scheduler():
    """Background scheduler for auto-capture processing."""
    schedule.every(30).seconds.do(
        process_pending_events, get_db_connection, add_memory_trace_logic
    )
    schedule.every().sunday.at("03:00").do(
        cleanup_old_auto_capture_events, get_db_connection, 30
    )

    while True:
        schedule.run_pending()
        time.sleep(1)

# Start background worker thread
auto_capture_thread = threading.Thread(target=run_auto_capture_scheduler, daemon=True)
auto_capture_thread.start()
logger.info(">>> [BOOT] Auto-Capture background worker iniciado.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

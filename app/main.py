"""
Aethera Cortex v1.0
Plataforma de Memória Soberana & Gestão de Contexto.
Multi-Tenant | Auth Dinâmica | Persistência Vetorial
"""

import sqlite3
import faiss
import numpy as np
import os
import time
import threading
import uuid
import secrets
import requests
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- CONFIGURAÇÃO ---
DATA_DIR = os.environ.get("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "aethera_cortex.db")
INDEX_PATH = os.path.join(DATA_DIR, "memory.index")

# Configuração de Modelo (Embedding Local)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aethera")

app = FastAPI(title="Aethera Cortex API")

# --- SINGLETONS & LOCKS ---
faiss_lock = threading.Lock()
logger.info(">>> [BOOT] Carregando Modelos Neurais...")
embed_model = SentenceTransformer(MODEL_NAME)
dim = embed_model.get_sentence_embedding_dimension()

# --- MODELOS PYDANTIC ---
class MemoryUpdate(BaseModel):
    content: str

class KeyRequest(BaseModel):
    owner_name: str
    tier: str = "free"  # 'root', 'pro', 'free'

# --- CAMADA DE DADOS (SQLITE + FAISS) ---

def init_db():
    """Inicializa o Schema Multi-Tenant do Aethera Cortex."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. Tabela de Memórias (Core)
    c.execute('''CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL,
                    vector_id INTEGER
                )''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)')

    # 2. Tabela de Autenticação (SaaS)
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
                    key TEXT PRIMARY KEY,
                    owner_name TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at REAL,
                    tier TEXT DEFAULT 'free'
                )''')

    # 3. Bootstrap: Geração da Chave Mestra (Se o banco estiver vazio)
    c.execute("SELECT count(*) FROM api_keys")
    if c.fetchone()[0] == 0:
        root_key = f"sk_aethera_root_{secrets.token_hex(16)}"
        logger.warning(f"\n{'='*60}")
        logger.warning(f" AETHERA CORTEX - PRIMEIRA EXECUÇÃO DETECTADA")
        logger.warning(f" CHAVE MESTRA (ROOT) GERADA: {root_key}")
        logger.warning(f" SALVE ESTA CHAVE IMEDIATAMENTE. ELA NÃO SERÁ MOSTRADA NOVAMENTE.")
        logger.warning(f"{'='*60}\n")
        
        c.execute("INSERT INTO api_keys (key, owner_name, is_active, created_at, tier) VALUES (?, ?, 1, ?, 'root')", 
                  (root_key, "System Administrator", time.time()))
        conn.commit()
    
    conn.close()

def load_index():
    if os.path.exists(INDEX_PATH):
        logger.info(">>> [BOOT] Carregando Índice Vetorial do Disco...")
        return faiss.read_index(INDEX_PATH)
    else:
        logger.info(">>> [BOOT] Criando Novo Índice Vetorial...")
        return faiss.IndexIDMap(faiss.IndexFlatL2(dim))

# Inicialização
init_db()
memory_index = load_index()

# --- SEGURANÇA (AUTH DINÂMICA) ---

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verifica se a chave existe no banco e está ativa."""
    if not api_key:
        raise HTTPException(status_code=403, detail="Acesso Negado: Header 'x-api-key' ausente.")
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT owner_name, tier FROM api_keys WHERE key = ? AND is_active = 1", (api_key,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return {"key": api_key, "owner": row[0], "tier": row[1]}
    
    # Delay artificial para evitar ataque de força bruta (Timing Attack Mitigation)
    time.sleep(0.1)
    raise HTTPException(status_code=403, detail="Aethera Security: Chave inválida ou revogada.")

# --- CORE LÓGICO ---

def add_memory_trace(session_id: str, role: str, content: str, background_tasks: BackgroundTasks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO memories (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, content, time.time()))
    row_id = c.lastrowid
    conn.commit()
    conn.close()

    def _vectorize_and_persist(text, db_id):
        try:
            vector = embed_model.encode([text])[0]
            vec_np = np.array([vector]).astype('float32')
            id_np = np.array([db_id]).astype('int64')

            with faiss_lock:
                memory_index.add_with_ids(vec_np, id_np)
                faiss.write_index(memory_index, INDEX_PATH)
                logger.info(f"DEBUG [MEMORY] Trace {db_id} persistido.")
        except Exception as e:
            logger.error(f"CRITICAL [MEMORY] Falha na vetorização: {e}")

    background_tasks.add_task(_vectorize_and_persist, content, row_id)
    return row_id

def retrieve_context(session_id: str, query: str, limit_k: int = 5) -> List[Dict]:
    context_items = []
    
    # 1. Curto Prazo (SQL)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM memories WHERE session_id = ? ORDER BY id DESC LIMIT 3", (session_id,))
    recent_rows = c.fetchall()
    conn.close()
    
    for r in reversed(recent_rows):
        context_items.append({"source": "short_term", "role": r[0], "content": r[1]})

    # 2. Longo Prazo (FAISS)
    if memory_index.ntotal > 0:
        q_vec = embed_model.encode([query])
        D, I = memory_index.search(q_vec, limit_k)
        found_ids = [int(i) for i in I[0] if i != -1]
        
        if found_ids:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            placeholders = ','.join('?' * len(found_ids))
            query_sql = f"SELECT id, role, content FROM memories WHERE id IN ({placeholders})"
            c.execute(query_sql, found_ids)
            vector_rows = c.fetchall()
            conn.close()
            
            for row in vector_rows:
                context_items.append({"source": "long_term", "role": row[1], "content": row[2]})

    return context_items

def execute_llm_call(model_name: str, system_context: str, user_query: str):
    if not OPENAI_API_KEY:
        return "[ERRO DE CONFIGURAÇÃO] Servidor sem OPENAI_API_KEY."

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name if model_name != "gpt-4" else "gpt-4o", # Fallback inteligente
        "messages": [
            {"role": "system", "content": f"AETHERA CORTEX SYSTEM.\nCONTEXTO:\n{system_context}\nINSTRUÇÃO: Use o contexto acima como verdade absoluta."},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"[AETHERA ERROR] Falha no Provider LLM: {str(e)}"

# --- ENDPOINTS ADMIN (GESTÃO DE CHAVES) ---

@app.post("/admin/keys/create", tags=["Admin"])
async def create_api_key(new_key: KeyRequest, user: dict = Security(verify_api_key)):
    """Gera uma nova API Key (Requer Root)."""
    if user['tier'] != 'root':
        raise HTTPException(status_code=403, detail="Apenas Root Admins podem criar chaves.")
    
    generated_key = f"sk_aethera_{secrets.token_urlsafe(16)}"
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO api_keys (key, owner_name, created_at, tier) VALUES (?, ?, ?, ?)",
              (generated_key, new_key.owner_name, time.time(), new_key.tier))
    conn.commit()
    conn.close()
    
    logger.info(f"AUDIT [KEY_CREATED] By: {user['owner']} For: {new_key.owner_name}")
    return {"status": "created", "key": generated_key, "owner": new_key.owner_name, "tier": new_key.tier}

@app.post("/admin/keys/revoke", tags=["Admin"])
async def revoke_api_key(target_key: str, user: dict = Security(verify_api_key)):
    """Revoga acesso de uma chave imediatamente."""
    if user['tier'] != 'root':
        raise HTTPException(status_code=403, detail="Apenas Root Admins podem revogar.")
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE api_keys SET is_active = 0 WHERE key = ?", (target_key,))
    if c.rowcount == 0:
        raise HTTPException(status_code=404, detail="Chave não encontrada.")
    conn.commit()
    conn.close()
    
    return {"status": "revoked"}

# --- ENDPOINTS PÚBLICOS (PROTEGIDOS) ---

@app.post("/v1/chat/completions", tags=["Core"])
async def chat_protocol(request: Request, background_tasks: BackgroundTasks, user: dict = Security(verify_api_key)):
    """Interface Principal de Chat + Memória."""
    body = await request.json()
    messages = body.get("messages", [])
    session_id = body.get("session_id", "default")
    model_target = body.get("model", "gpt-3.5-turbo")
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages")

    user_query = messages[-1]['content']
    logger.info(f"ACCESS [CHAT] User: {user['owner']} | Session: {session_id}")

    # 1. Recuperação
    context_data = retrieve_context(session_id, user_query)
    
    # 2. Memory-Only Bypass (Ingestão Gratuita)
    if model_target == "memory-only":
        add_memory_trace(session_id, "user", user_query, background_tasks)
        return JSONResponse({
            "id": f"mem-{uuid.uuid4()}",
            "choices": [{"message": {"role": "assistant", "content": "[AETHERA] Memória ingerida com sucesso."}}]
        })

    # 3. Montagem de Prompt
    system_instruction = ""
    for item in context_data:
        system_instruction += f"- [{item['source'].upper()}]: {item['content']}\n"
    
    # 4. Execução
    ai_response = execute_llm_call(model_target, system_instruction, user_query)

    # 5. Persistência
    add_memory_trace(session_id, "user", user_query, background_tasks)
    add_memory_trace(session_id, "assistant", ai_response, background_tasks)

    return JSONResponse({
        "id": str(uuid.uuid4()),
        "model": model_target,
        "choices": [{"message": {"role": "assistant", "content": ai_response}, "finish_reason": "stop"}]
    })

# --- CRUD DE MEMÓRIAS ---

@app.get("/v1/memories", tags=["Management"])
async def list_memories(limit: int = 10, offset: int = 0, user: dict = Security(verify_api_key)):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, role, content, timestamp, session_id FROM memories ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"data": rows}

@app.delete("/v1/memories/{memory_id}", tags=["Management"])
async def delete_memory(memory_id: int, user: dict = Security(verify_api_key)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    if deleted == 0: raise HTTPException(404, "Memory ID not found")
    return {"status": "deleted", "id": memory_id}

@app.put("/v1/memories/{memory_id}", tags=["Management"])
async def update_memory(memory_id: int, update: MemoryUpdate, user: dict = Security(verify_api_key)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE memories SET content = ? WHERE id = ?", (update.content, memory_id))
    updated = c.rowcount
    conn.commit()
    conn.close()
    if updated == 0: raise HTTPException(404, "Memory ID not found")
    return {"status": "updated", "id": memory_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

"""
Universal Memory Gateway.
Transforma LLMs em processadores stateless. A memória vive aqui.
"""

import sqlite3
import faiss
import numpy as np
import os
import time
import threading
import uuid
import requests
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pydantic import BaseModel

class MemoryUpdate(BaseModel):
    content: str

# Carrega variáveis de ambiente do arquivo .env se existir
load_dotenv()

# --- CONFIGURAÇÃO ---
DATA_DIR = os.environ.get("DATA_DIR", "memory_data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "history.db")
INDEX_PATH = os.path.join(DATA_DIR, "memory.index")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Modelo rápido para indexação de memória (não precisa de Cross-Encoder aqui ainda)
# Usamos o MiniLM para varrer grandes históricos rapidamente.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Sovereign Memory Gateway")

# --- SINGLETONS & LOCKS ---
faiss_lock = threading.Lock()
# Initialize model globally
embed_model = SentenceTransformer(MODEL_NAME)
dim = embed_model.get_sentence_embedding_dimension()

# --- CAMADA DE DADOS (SQLITE + FAISS) ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Schema focado em Sessão e Histórico Linear
    c.execute('''CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL,
                    vector_id INTEGER  -- Link para o FAISS
                )''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)')
    conn.commit()
    conn.close()

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    else:
        # IndexIDMap para garantir consistência entre SQLite ID e FAISS ID
        return faiss.IndexIDMap(faiss.IndexFlatL2(dim))

# Inicialização Global
init_db()
memory_index = load_index()

# --- FUNÇÕES CORE DE MEMÓRIA ---

def add_memory_trace(session_id: str, role: str, content: str, background_tasks: BackgroundTasks):
    """
    Grava um pensamento/fala no 'Cérebro'.
    1. Salva texto no SQLite.
    2. Gera vetor e salva no FAISS (Async).
    """
    # 1. SQLite (Imediato para garantir consistência de dados)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO memories (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, content, time.time()))
    row_id = c.lastrowid
    conn.commit()
    conn.close()

    # 2. FAISS (Vetorização e Escrita)
    # Função interna para rodar em background/thread segura
    def _vectorize_and_persist(text, db_id):
        vector = embed_model.encode([text])[0]
        vec_np = np.array([vector]).astype('float32')
        id_np = np.array([db_id]).astype('int64')

        with faiss_lock:
            memory_index.add_with_ids(vec_np, id_np)
            # Update no SQLite com o ID do vetor (opcional, mas bom para integridade)
            # Persiste disco
            faiss.write_index(memory_index, INDEX_PATH)
            print(f"DEBUG [MEMORY] Trace {db_id} vetorizado e persistido.")

    background_tasks.add_task(_vectorize_and_persist, content, row_id)
    return row_id

def retrieve_context(session_id: str, query: str, limit_k: int = 5) -> List[Dict]:
    """
    Recupera memórias relevantes para a query atual.
    Estratégia Híbrida:
    1. Memória de Curto Prazo: Últimas N mensagens da sessão (SQLite).
    2. Memória de Longo Prazo: Busca vetorial no FAISS (RAG).
    """
    context_items = []
    
    # A. Curto Prazo (Recency Bias) - Pega as últimas 3 interações
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM memories WHERE session_id = ? ORDER BY id DESC LIMIT 3", (session_id,))
    recent_rows = c.fetchall()
    conn.close()
    
    # Inverte para ordem cronológica porque pegamos DESC
    for r in reversed(recent_rows):
        context_items.append({"source": "short_term", "role": r[0], "content": r[1]})

    # B. Longo Prazo (Semantic Search)
    if memory_index.ntotal > 0:
        q_vec = embed_model.encode([query])
        # Search FAISS
        D, I = memory_index.search(q_vec, limit_k)
        
        found_ids = [int(i) for i in I[0] if i != -1]
        
        if found_ids:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            placeholders = ','.join('?' * len(found_ids))
            
            # Trazemos o conteúdo. 
            # Nota: Isso pode trazer redundância com o short_term se a mensagem recente for semanticamente relevante.
            # Para um MVP está ótimo, o LLM geralmente sabe lidar com repetição no contexto.
            query_sql = f"SELECT id, role, content FROM memories WHERE id IN ({placeholders})"
            c.execute(query_sql, found_ids)
            vector_rows = c.fetchall()
            conn.close()
            
            for row in vector_rows:
                context_items.append({"source": "long_term", "role": row[1], "content": row[2]})

    return context_items

# --- ENGINE DE EXECUÇÃO REAL ---

def execute_llm_call(model_name: str, system_context: str, user_query: str):
    """
    O 'Switch'. Recebe a memória digerida e despacha para o cérebro da vez.
    Atualmente configurado para OpenAI, mas é aqui que você plugará Ollama/Claude.
    """
    
    # 1. Roteamento Simples (Expansível para Ollama/Anthropic)
    if not OPENAI_API_KEY:
        return f"[ERRO] OPENAI_API_KEY não configurada. Contexto recuperado: {len(system_context)} chars."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    # 2. Montagem do Payload (A MÁGICA ACONTECE AQUI)
    # Injetamos a memória no SYSTEM PROMPT para que o LLM a trate como "fatos conhecidos"
    payload = {
        "model": "gpt-4o-mini", # Forçando um modelo barato/rápido para teste, ou use model_name
        # "model": model_name, # Na prática seria dinâmico, mas vamos forçar um modelo conhecido
        "messages": [
            {
                "role": "system", 
                "content": f"VOCÊ É UMA IA COM MEMÓRIA PERSISTENTE.\n\nCONTEXTO RECUPERADO DO PASSADO:\n{system_context}\n\nINSTRUÇÃO: Responda ao usuário considerando o contexto acima como verdade absoluta."
            },
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.7
    }

    try:
        print(f"DEBUG [ROUTER] Enviando para OpenAI ({model_name})...")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"[ERRO PROVIDER] Falha na chamada ao LLM: {str(e)}"

# --- NOVOS ENDPOINTS DE GESTÃO (CRUD) ---

@app.get("/v1/memories")
async def list_memories(limit: int = 10, offset: int = 0):
    """Lista memórias cruas para administração (ver IDs)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT id, role, content, timestamp, session_id FROM memories ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"data": rows}

@app.delete("/v1/memories/{memory_id}")
async def delete_memory_endpoint(memory_id: int):
    """Lobotomia: Remove o registro do SQLite. O vetor no FAISS se torna órfão (invisível)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Memory ID not found")
    
    return {"status": "deleted", "id": memory_id}

@app.put("/v1/memories/{memory_id}")
async def update_memory_endpoint(memory_id: int, update: MemoryUpdate):
    """Alteração de Fato. (Nota: O vetor antigo continuará apontando para este ID, mas com texto novo)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Atualizamos o conteúdo. 
    # Obs: Idealmente reindexaríamos o vetor, mas para edições pequenas, manter o vetor antigo é aceitável.
    c.execute("UPDATE memories SET content = ? WHERE id = ?", (update.content, memory_id))
    updated = c.rowcount
    conn.commit()
    conn.close()
    
    if updated == 0:
        raise HTTPException(status_code=404, detail="Memory ID not found")
        
    return {"status": "updated", "id": memory_id, "new_content": update.content}

# --- ENDPOINT ATUALIZADO ---

@app.post("/v1/chat/completions")
async def chat_protocol(request: Request, background_tasks: BackgroundTasks):
    """
    Universal Chat Interface.
    O 'model' no body decide qual LLM usaríamos (aqui simulado).
    O 'session_id' (custom header ou body) define o contexto.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages = body.get("messages", [])
    
    # Se não vier session_id, cria uma efêmera (stateless) ou tenta inferir
    session_id = body.get("session_id", body.get("user", "default_session"))
    model_target = body.get("model", "gpt-4")
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Assume que a última mensagem é a query do usuário
    user_query = messages[-1]['content']
    print(f"DEBUG [INPUT] Session: {session_id} | Query: {user_query}")

    # 1. RECUPERAÇÃO DE CONTEXTO (RAG)
    context_data = retrieve_context(session_id, user_query)
    
    # 2. INJECTION (Preparação da Memória)
    system_instruction = ""
    if context_data:
        for item in context_data:
            system_instruction += f"- [{item['role'].upper()}]: {item['content']}\n"
    else:
        system_instruction = "Nenhuma memória relevante encontrada."

    # [BYPASS] Se o modelo for "memory-only", salvamos mas não chamamos a OpenAI
    if model_target == "memory-only":
        print(f"DEBUG [BYPASS] Modo ingestão pura. Memória salva, LLM ignorado.")
        # Persistência (Grava apenas o User input)
        add_memory_trace(session_id, "user", user_query, background_tasks)
        
        return JSONResponse({
            "id": "mem-only-" + str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "memory-only",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "[SISTEMA] Memória ingerida com sucesso. LLM não acionado."},
                "finish_reason": "stop"
            }]
        })
    
    # 3. EXECUÇÃO REAL (Chamada ao LLM)
    # Passamos o execute_llm_call
    ai_response = execute_llm_call(model_target, system_instruction, user_query)

    # 4. PERSISTÊNCIA (Gravação do Turno)
    # Salva o que o usuário disse
    add_memory_trace(session_id, "user", user_query, background_tasks)
    # Salva o que a IA respondeu
    add_memory_trace(session_id, "assistant", ai_response, background_tasks)

    return JSONResponse({
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_target,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ai_response},
            "finish_reason": "stop"
        }]
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

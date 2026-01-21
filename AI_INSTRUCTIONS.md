# AI Instructions - Aethera Cortex Development Guide

Este documento contém **regras operacionais** para agentes de IA modificarem o projeto com segurança.

---

## 🎯 Objetivo

Permitir que agentes de IA façam modificações consistentes, seguras e testáveis no Aethera Cortex, seguindo os padrões estabelecidos.

---

## 📋 Como Adicionar Novos Componentes

### 1. Novo Endpoint REST

**Localização:** `app/main.py` (após linha ~920)

**Template:**
```python
@app.{method}("/v1/{resource}", tags=["{Tag}"])
async def {operation}_{resource}(
    {params},
    user: dict = Security(verify_api_key)
):
    """
    {Descrição do endpoint}
    
    Args:
        {param_docs}
    
    Returns:
        JSON response with {schema}
    
    Raises:
        HTTPException: 403 (auth), 404 (not found), 429 (rate limit)
    """
    # 1. Validar input
    if not {validation}:
        raise HTTPException(400, "Invalid input")
    
    # 2. Business logic
    conn = get_db_connection()
    c = conn.cursor()
    # ... SQL queries
    conn.commit()
    conn.close()
    
    # 3. Return response
    return {"status": "success", "data": result}
```

**Checklist:**
- [ ] Add `Security(verify_api_key)` para auth
- [ ] Adicionar docstring com Args/Returns/Raises
- [ ] Tag apropriada (`Core`, `Admin`, `Usage`)
- [ ] Status codes semânticos (200, 404, 403, 429, 500)
- [ ] Commit + close connection no finally
- [ ] Testar com `curl` ou `requests`

---

### 2. Nova Ferramenta MCP

**Localização:** `app/main.py` (seção `@mcp_server.list_tools()` linha ~410)

**Step 1: Adicionar Schema**
```python
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ... existing tools
        Tool(
            name="{tool_name}",
            description="{O que a ferramenta faz}",
            inputSchema={
                "type": "object",
                "properties": {
                    "{param}": {
                        "type": "{string|integer|boolean}",
                        "description": "{Descrição do parâmetro}"
                    }
                },
                "required": ["{required_params}"]
            }
        )
    ]
```

**Step 2: Implementar Handler**
```python
@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # ... existing tools
    
    elif name == "{tool_name}":
        param = arguments.get("{param}")
        try:
            # Business logic
            result = do_something(param)
            return [TextContent(type="text", text=f"Success: {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Erro: {e}")]
```

**Step 3: Testar**
```bash
# Usar Claude Desktop ou inspect_mcp.py
python inspect_mcp.py
```

**Checklist:**
- [ ] Schema no `list_tools()`
- [ ] Handler no `call_tool()`
- [ ] Documentação em português
- [ ] Error handling com try/except
- [ ] Retornar sempre `list[TextContent]`
- [ ] Testar com MCP client real

---

### 3. Novo Provider/Integration

**Localização:** Criar novo arquivo `app/{provider}_adapter.py`

**Template:**
```python
"""
Adapter para integração com {Provider}.
"""

import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger("Aethera.{Provider}")

class {Provider}Adapter:
    """
    Cliente para API do {Provider}.
    
    Attributes:
        api_key: API key do provider
        base_url: URL base da API
    """
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def {method_name}(self, params: Dict) -> Optional[Dict]:
        """
        {Descrição do método}
        
        Args:
            params: Parâmetros da requisição
        
        Returns:
            Response data ou None em erro
        
        Raises:
            requests.HTTPError: Em caso de erro HTTP
        """
        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                json=params,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"[{Provider.upper()}] Error: {e}")
            return None
```

**Integração no main.py:**
```python
# No topo do arquivo
from app.{provider}_adapter import {Provider}Adapter

# Após init_db()
{provider}_client = None
if os.environ.get("{PROVIDER}_API_KEY"):
    {provider}_client = {Provider}Adapter(
        api_key=os.environ["{PROVIDER}_API_KEY"],
        base_url=os.environ.get("{PROVIDER}_URL", "https://...")
    )
    logger.info(">>> [BOOT] {Provider} initialized.")
```

**Checklist:**
- [ ] Criar arquivo separado em `app/`
- [ ] Docstrings completas (Google style)
- [ ] Logging com prefixo `[PROVIDER]`
- [ ] Timeout em requisições HTTP (30s padrão)
- [ ] Error handling robusto
- [ ] Env vars documentadas
- [ ] Inicialização condicional (graceful degradation)

---

### 4. Novo Worker/Background Job

**Localização:** `app/main.py` (após middlewares, linha ~330)

**Template:**
```python
import threading
import schedule
import time

def {job_function}():
    """
    {Descrição do job}
    Executado a cada {interval}.
    """
    logger.info(f"[JOB] {Job Name} starting...")
    try:
        conn = get_db_connection()
        c = conn.cursor()
        # ... lógica do job
        conn.commit()
        conn.close()
        logger.info(f"[JOB] {Job Name} completed.")
    except Exception as e:
        logger.error(f"[JOB] {Job Name} failed: {e}")

# Agendar job
def run_scheduler():
    schedule.every().{interval}.do({job_function})
    while True:
        schedule.run_pending()
        time.sleep(60)

# Iniciar thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()
logger.info(">>> [BOOT] Background jobs scheduled.")
```

**Checklist:**
- [ ] Thread daemon=True (não bloqueia shutdown)
- [ ] Error handling com try/except + logging
- [ ] Connection management (open/close)
- [ ] Intervalo apropriado (evitar sobrecarga)
- [ ] Log início e fim de execução
- [ ] Adicionar ao `requirements.txt`: `schedule`

---

## 🔒 Padrões Obrigatórios do Repo

### 1. Typing
```python
# ✅ CORRETO
def retrieve_context(session_id: str, query: str, limit: int = 5) -> List[Dict]:
    ...

# ❌ ERRADO
def retrieve_context(session_id, query, limit=5):
    ...
```

### 2. Async/Await
```python
# ✅ CORRETO - Endpoints FastAPI
@app.get("/v1/memories")
async def list_memories(...):
    ...

# ✅ CORRETO - Lógica síncrona separada
def add_memory_trace_logic(session_id: str, ...):
    # Código síncrono com psycopg2
    ...
```

### 3. Logging
```python
# ✅ CORRETO
logger.info(f"[CONTEXT] Processing query: {query[:50]}...")
logger.warning(f"[RATE_LIMITER] Limit exceeded: {api_key[:20]}...")
logger.error(f"CRITICAL [DB] Connection failed: {e}")

# ❌ ERRADO
print("Processing query")  # Não usar print()
```

### 4. Validação
```python
# ✅ CORRETO - Pydantic Models
class MemoryUpdate(BaseModel):
    content: str

@app.put("/v1/memories/{id}")
async def update(id: int, update: MemoryUpdate, ...):
    ...

# ❌ ERRADO - Raw dict sem validação
@app.put("/v1/memories/{id}")
async def update(id: int, request: Request):
    data = await request.json()  # Sem validação
```

### 5. Error Handling
```python
# ✅ CORRETO
try:
    conn = get_db_connection()
    # ... operações
    conn.commit()
except Exception as e:
    logger.error(f"DB Error: {e}")
    raise HTTPException(500, "Internal error")
finally:
    if conn:
        conn.close()

# ❌ ERRADO - Sem tratamento
conn = get_db_connection()
c.execute("...")  # Pode falhar sem catch
```

### 6. Security
```python
# ✅ CORRETO - Auth obrigatória
@app.get("/v1/sensitive")
async def endpoint(user: dict = Security(verify_api_key)):
    ...

# ❌ ERRADO - Endpoint desprotegido
@app.get("/v1/sensitive")
async def endpoint():
    ...
```

---

## ✅ Checklist de PR

### Antes de Commitar

- [ ] **Lint**: Código segue PEP 8
- [ ] **Type Hints**: Todas funções públicas tipadas
- [ ] **Docstrings**: Funções complexas documentadas (Google style)
- [ ] **Logging**: Ações importantes logadas com prefixo
- [ ] **Error Handling**: Try/except em I/O e external calls
- [ ] **Tests**: Adicionar teste para feature nova
- [ ] **Secrets**: NUNCA commitar API keys ou senhas
- [ ] **Migrations**: Se mudou schema, atualizar `init_db()`

### Comandos Pré-Commit

```bash
# 1. Lint (manual)
# TODO: Adicionar flake8 ou black ao projeto

# 2. Run tests
python tests/test_auth.py
python tests/test_crud.py
python tests/test_rate_limits.py

# 3. Manual test
curl -H "x-api-key: sk_aethera_..." http://localhost:8001/v1/memories
```

### Durante Code Review

- [ ] Código é self-explanatory ou tem comentários
- [ ] Mudanças são **mínimas** para resolver o problema
- [ ] Sem código comentado ou dead code
- [ ] Variáveis têm nomes descritivos
- [ ] SQL injection prevenido (usar parametrized queries)
- [ ] Rate limiting aplicado a endpoints novos

---

## 🚫 Nunca Faça (Anti-Patterns)

### 1. ❌ Executar SQL com String Interpolation
```python
# ERRADO - SQL Injection vulnerability
c.execute(f"SELECT * FROM memories WHERE id = {memory_id}")

# CORRETO - Parametrized query
c.execute("SELECT * FROM memories WHERE id = %s", (memory_id,))
```

### 2. ❌ Expor Chaves ou Dados Sensíveis em Logs
```python
# ERRADO
logger.info(f"User logged in: {api_key}")

# CORRETO - Mascarar
logger.info(f"User logged in: {api_key[:20]}...")
```

### 3. ❌ Ignorar Erros Silenciosamente
```python
# ERRADO
try:
    critical_operation()
except:
    pass  # Silencia erro crítico

# CORRETO
try:
    critical_operation()
except Exception as e:
    logger.error(f"Critical operation failed: {e}")
    raise HTTPException(500, "Internal error")
```

### 4. ❌ Hardcoded Configs
```python
# ERRADO
DATABASE_URL = "postgresql://user:pass@localhost/db"

# CORRETO
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set")
```

### 5. ❌ Blocar Event Loop com I/O Síncrono
```python
# ERRADO em endpoint async
@app.get("/slow")
async def slow():
    time.sleep(5)  # Bloqueia event loop
    return {"done": True}

# CORRETO - Use background tasks
@app.get("/slow")
async def slow(background_tasks: BackgroundTasks):
    background_tasks.add_task(slow_operation)
    return {"status": "processing"}
```

### 6. ❌ Criar Endpoints sem Auth
```python
# ERRADO - Aberto publicamente
@app.delete("/v1/memories/{id}")
async def delete_memory(id: int):
    ...

# CORRETO
@app.delete("/v1/memories/{id}")
async def delete_memory(id: int, user: dict = Security(verify_api_key)):
    ...
```

---

## 🗺️ Mapa de Navegação do Fluxo Principal

### Fluxo 1: Cliente REST → Memória Persistida

```
1. Request chega: POST /v1/chat/completions
   ↓
2. Middleware: RateLimitMiddleware
   → Valida rate limit do tier
   → Registra usage log
   ↓
3. Middleware: McpAuthMiddleware (apenas SSE)
   ↓
4. Endpoint: chat_protocol() [app/main.py:858]
   ↓
5. retrieve_context_logic() [app/main.py:350]
   → SELECT recentes (short-term)
   → SELECT similares via pgvector (long-term)
   ↓
6. execute_llm_call() [app/main.py:381]
   → POST para OpenAI API
   ↓
7. add_memory_trace() [app/main.py:378]
   → Background task
   → embed_model.encode()
   → INSERT INTO memories
   ↓
8. Response: {"choices": [...]}
```

### Fluxo 2: Claude Desktop (MCP) → Recall

```
1. Claude invoca tool "recall" via stdio
   ↓
2. mcp_server.py recebe via FastMCP [linha 45]
   ↓
3. brain.ask() via client.py SDK [linha 58]
   ↓
4. POST http://backend:8001/v1/chat/completions
   ↓
5. [Mesmo fluxo do REST acima]
   ↓
6. Response volta via stdio para Claude
```

### Fluxo 3: Admin Cria Nova API Key

```
1. POST /admin/keys/create
   ↓
2. verify_api_key() [app/main.py:191]
   → Valida que user.tier == 'root'
   ↓
3. Gera nova key: sk_aethera_{random}
   ↓
4. INSERT INTO api_keys
   ↓
5. Response: {"key": "sk_aethera_..."}
```

---

## 🔧 Config e Environment Variables

### Variáveis Obrigatórias

```bash
# .env (criar baseado neste template)

# Database (Neon ou PostgreSQL local)
DATABASE_URL="postgresql://user:password@host:5432/dbname?sslmode=require"

# OpenAI (para LLM calls)
OPENAI_API_KEY="sk-proj-..."

# Opcional
DB_SCHEMA="public"  # ou "test" para ambiente de teste
MCP_PUBLIC_URL="http://localhost:8001"  # URL pública do servidor
```

### Como Usar

```python
# No código
import os
from dotenv import load_dotenv

load_dotenv()  # Carrega .env

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
```

### ⚠️ Segurança

- **NUNCA** commitar `.env` no git
- Adicionar `.env` no `.gitignore`
- Usar secrets manager em produção (AWS Secrets, Vault)
- Rotar keys regularmente
- Validar HTTPS em produção (`sslmode=require`)

---

## 🧪 Testing Obrigatório Antes de Commit

### 1. Test Suite Básico

```bash
# Subir servidor local
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Em outro terminal
python tests/test_auth.py      # Auth + Security
python tests/test_crud.py       # CRUD operations
python tests/test_rate_limits.py  # Rate limiting
```

**Mínimo Aceitável:**
- ✅ Todos testes passam (prints mostram `[PASS]`)
- ✅ Nenhum erro 500 não tratado
- ✅ Rate limits funcionam (429 quando apropriado)

### 2. Smoke Test Manual

```bash
# 1. Health check
curl http://localhost:8001/

# 2. List memories (requer API key válida)
curl -H "x-api-key: sk_aethera_..." \
     http://localhost:8001/v1/memories

# 3. Create memory
curl -X POST http://localhost:8001/v1/chat/completions \
     -H "x-api-key: sk_aethera_..." \
     -H "Content-Type: application/json" \
     -d '{"model":"memory-only","session_id":"test","messages":[{"role":"user","content":"Test memory"}]}'

# 4. Check rate limit
curl http://localhost:8001/v1/usage \
     -H "x-api-key: sk_aethera_..."
```

### 3. MCP Integration Test

```bash
# Testar MCP tools
python inspect_mcp.py

# Ou via Claude Desktop (config em .roo/mcp.json)
# Perguntar: "Lembre que meu café favorito é espresso"
# Depois: "Qual é meu café favorito?"
```

### 4. Performance Baseline (Opcional)

```bash
# Usando teste-de-validacao/attack.py
python teste-de-validacao/attack.py
```

**Metas:**
- 🎯 p50 < 50ms (RAG query)
- 🎯 p95 < 200ms
- 🎯  100 req/s sem erros (hardware local modesto)

---

## 📊 Observability & Debugging

### Logs Importantes

```bash
# Seguir logs em tempo real
tail -f /var/log/aethera.log  # Produção
# ou
docker logs -f aethera-cortex  # Docker

# Filtrar por módulo
grep "\[RATE_LIMITER\]" logs.txt
grep "CRITICAL" logs.txt
```

### Endpoints de Debug

```python
# Adicionar temporariamente para debug
@app.get("/debug/config")
async def debug_config():
    return {
        "db_schema": DB_SCHEMA,
        "has_openai_key": bool(OPENAI_API_KEY),
        "model_dim": dim
    }
```

**⚠️ REMOVER antes de produção!**

### Common Issues

| Sintoma | Causa Provável | Solução |
|---------|----------------|---------|
| 403 Forbidden | API key inválida | Verificar header/query param |
| 429 Too Many Requests | Rate limit | Upgrade tier ou esperar reset |
| 500 Internal Error | DB connection | Verificar DATABASE_URL |
| Embedding lento | CPU fraco | Reduzir batch size ou usar GPU |
| Memory leak | Connections não fechadas | Adicionar finally: conn.close() |

---

## 🔄 Workflow Recomendado

### Feature Nova

```bash
# 1. Branch
git checkout -b feature/nova-feature

# 2. Desenvolver
# ... código

# 3. Testar localmente
python tests/test_*.py

# 4. Commit
git add .
git commit -m "feat: adiciona nova feature"

# 5. Push
git push origin feature/nova-feature

# 6. PR + Code Review
```

### Hotfix em Produção

```bash
# 1. Branch a partir de main
git checkout -b hotfix/critical-bug

# 2. Fix mínimo
# ... código

# 3. Test + Deploy rápido
python tests/test_auth.py  # Crítico
docker build -t aethera:hotfix .
docker run ... # Testar container

# 4. Merge ASAP
git push origin hotfix/critical-bug
# PR direto para main
```

---

## 📚 Recursos Adicionais

- **Codebase Principal**: `app/main.py` - Ler top-to-bottom
- **SDK Usage**: `client.py` - Exemplos de uso
- **MCP Integration**: `mcp_server.py` - Standalone server
- **Test Examples**: `tests/` - Casos de uso reais

---

_Última atualização: 2026-01-21 | Aethera Cortex v2.1_

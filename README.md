# 🧠 Aethera Cortex v2.1

**Plataforma de Memória Soberana & Gestão de Contexto para Agentes de IA**

[![Status](https://img.shields.io/badge/status-production--ready-brightgreen)](.)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](.)
[![FastAPI](https://img.shields.io/badge/FastAPI-modern-009688)](.)
[![MCP](https://img.shields.io/badge/MCP-2025--03--26-purple)](https://spec.modelcontextprotocol.io)

---

## 🎯 O Que É?

Aethera Cortex permite que **agentes de IA mantenham memória de longo prazo** usando:
- 🧬 **Embeddings Vetoriais** (SentenceTransformers)
- 🔍 **Busca Semântica** (pgvector)
- 🔐 **Multi-Tenancy** (API keys + rate limiting)
- 🔌 **MCP Protocol** (Claude Desktop, IDEs)
- ☁️ **Cloud Native** (PostgreSQL/Neon, Docker)

**Use Cases:**
- Claude Desktop com memória persistente
- Chatbots com contexto histórico
- Assistentes personalizados com preferências do usuário
- Sistemas RAG (Retrieval-Augmented Generation)

---

## 📚 Documentação

Este repositório possui **documentação padronizada em 4 níveis**:

### 📖 Nível 1: Arquitetura Principal
**[ARCHITECTURE.md](./ARCHITECTURE.md)** - Documento principal (568 linhas)
- Propósito e objetivos do projeto
- Diagramas Mermaid (flowcharts + componentes)
- Modelo de dados detalhado
- Stack tecnológica completa
- ADRs (decisões arquiteturais)
- Roadmap e TODOs

### 🤖 Nível 2: Guia para Agentes de IA
**[AI_INSTRUCTIONS.md](./AI_INSTRUCTIONS.md)** - Regras operacionais (726 linhas)
- Como adicionar endpoints, tools, integrações
- Padrões obrigatórios (typing, async, logging)
- Checklist de PR e testes
- Anti-patterns e segurança
- Workflows recomendados

### 📦 Nível 3: READMEs por Módulo
- **[app/README.md](./app/README.md)** - Core application (422 linhas)
- **[tests/README.md](./tests/README.md)** - Test suite (499 linhas)

Cada README contém:
- Propósito do módulo
- Principais arquivos e interfaces
- Fluxos e diagramas
- Exemplos de uso práticos
- Pontos de atenção

### 💬 Nível 4: Documentação Inline
**[INLINE_DOCS.md](./INLINE_DOCS.md)** - Top 15 pontos + 5 docstrings (540 linhas)
- Lista priorizada de onde adicionar docstrings
- 5 exemplos completos prontos para copiar
- Google-style Python docstrings

### 🗂️ Extra: Contexto Compacto
**[ai-context.toon](./ai-context.toon)** - TOON format (233 linhas)
- Arquivo ultra-compacto para IAs
- Todas seções principais em formato estruturado
- Entry points, stack, data model, tests

---

## 🚀 Quick Start

### Pré-requisitos
- Python 3.10+
- PostgreSQL com pgvector
- OpenAI API key (para LLM calls)

### 1. Instalação

```bash
# Clone
git clone https://github.com/phfarath/uni-memory.git
cd uni-memory

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuração

```bash
# Crie .env (ou export direto)
cat > .env << EOF
DATABASE_URL="postgresql://user:pass@host:5432/dbname?sslmode=require"
OPENAI_API_KEY="sk-proj-..."
EOF
```

### 3. Run

```bash
# Local
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Ou com Docker
docker-compose up -d
```

### 4. Teste

```bash
# Health check
curl http://localhost:8001/

# Criar API key (primeira execução gera ROOT key, checar logs)
export ROOT_KEY="sk_aethera_root_..."

curl -X POST http://localhost:8001/admin/keys/create \
  -H "x-api-key: $ROOT_KEY" \
  -H "Content-Type: application/json" \
  -d '{"owner_name": "teste", "tier": "free"}'

# Testar memória
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "x-api-key: sk_aethera_..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "memory-only",
    "session_id": "test",
    "messages": [{"role": "user", "content": "Meu nome é João"}]
  }'
```

---

## 🧪 Testes

```bash
# Subir servidor
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Run test suite
python tests/test_auth.py         # Auth + Security
python tests/test_crud.py          # Memory CRUD
python tests/test_rate_limits.py  # Rate limiting (requer ROOT_KEY)
python tests/test_sdk.py           # SDK integration
```

**Ver:** [tests/README.md](./tests/README.md) para detalhes.

---

## 🔌 Integração com Claude Desktop

### 1. Configurar MCP Server

**Opção A: HTTP Transport (Recomendado)**

Adicione em `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "aethera-cortex": {
      "type": "streamable-http",
      "url": "http://localhost:8001/mcp",
      "headers": {
        "x-api-key": "sk_aethera_YOUR_KEY"
      }
    }
  }
}
```

**Opção B: SSE Transport (Legacy)**

```json
{
  "mcpServers": {
    "aethera-cortex": {
      "type": "sse",
      "url": "http://localhost:8001/mcp/sse?x-api-key=sk_aethera_YOUR_KEY"
    }
  }
}
```

### 2. Ferramentas Disponíveis

Claude Desktop terá acesso a 5 tools:
- 📝 **remember**: Gravar memória
- 🔍 **recall**: Buscar memórias relevantes
- 📋 **list_recent**: Listar memórias recentes
- ✏️ **update_memory**: Atualizar memória existente
- 🗑️ **forget**: Deletar memória

### 3. Exemplo de Uso

```
User: "Lembre que meu café favorito é cappuccino"
Claude: [usa tool remember] ✅ Memória gravada!

User: "Qual é meu café preferido?"
Claude: [usa tool recall] Seu café favorito é cappuccino!
```

---

## 🏗️ Arquitetura

```
Cliente (Claude/REST) 
    ↓
FastAPI Gateway (port 8001)
    ↓
[Auth Middleware] → [Rate Limit Middleware]
    ↓
Endpoint Handler
    ↓
RAG Pipeline:
  - SentenceTransformer (embedding)
  - PostgreSQL + pgvector (search)
  - OpenAI API (synthesis)
    ↓
Background Tasks (persistence)
    ↓
PostgreSQL (Neon)
```

**Ver:** [ARCHITECTURE.md](./ARCHITECTURE.md) para diagramas completos.

---

## 📊 Features

✅ **Implementadas:**
- Autenticação via API keys
- Rate limiting por tier (free, pro, team, root)
- Memory CRUD (create, read, update, delete)
- RAG pipeline completo
- MCP protocol (SSE + Streamable HTTP)
- Admin endpoints (keys, tiers, stats)
- Usage tracking e reporting
- Docker deployment

🚧 **Roadmap:**
- `.env.example` documentado
- Multi-tenant memory isolation (FK api_key)
- CI/CD pipeline
- Frontend dashboard
- API key rotation
- Webhook support

---

## 🛠️ Stack

**Backend:**
- FastAPI + Uvicorn (async REST API)
- PostgreSQL + pgvector (vector database)
- psycopg2 (database adapter)

**AI/ML:**
- SentenceTransformers (all-MiniLM-L6-v2)
- OpenAI API (GPT-3.5/4)

**Integration:**
- MCP SDK v1.3.0 (Model Context Protocol)
- aiohttp (async HTTP client)

**DevOps:**
- Docker + docker-compose
- Neon.tech (managed PostgreSQL)

---

## 📝 Como Contribuir

### 1. Para Desenvolvedores

1. Leia **[ARCHITECTURE.md](./ARCHITECTURE.md)** para entender o sistema
2. Leia **[AI_INSTRUCTIONS.md](./AI_INSTRUCTIONS.md)** para padrões de código
3. Faça suas mudanças seguindo as convenções
4. Rode test suite: `python tests/test_*.py`
5. Abra PR com descrição clara

### 2. Para Agentes de IA

Use **[AI_INSTRUCTIONS.md](./AI_INSTRUCTIONS.md)** como referência operacional:
- Templates para novos componentes
- Padrões obrigatórios
- Checklist de PR
- Anti-patterns a evitar

---

## 📄 Licença

TODO: Adicionar licença

---

## 🤝 Suporte

- **Issues**: [GitHub Issues](https://github.com/phfarath/uni-memory/issues)
- **Docs**: Ver arquivos de documentação listados acima
- **MCP Spec**: https://spec.modelcontextprotocol.io

---

## 🎓 Recursos

- [FastAPI Docs](https://fastapi.tiangolo.com)
- [pgvector Docs](https://github.com/pgvector/pgvector)
- [MCP Specification](https://spec.modelcontextprotocol.io)
- [SentenceTransformers](https://www.sbert.net)
- [Neon PostgreSQL](https://neon.tech)

---

**Última atualização:** 2026-01-21  
**Versão:** 2.1  
**Status:** Production-Ready ✅

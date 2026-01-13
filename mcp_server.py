"""
ColdBreaker Memory MCP Server.
Expõe o Cérebro Soberano (Docker) como ferramentas nativas para o Claude/IDEs.
"""

from mcp.server.fastmcp import FastMCP
from client import SovereignBrain
import logging

# Configuração de Logs (Essencial para Produtização)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-memory")

# Inicializa o Servidor MCP
mcp = FastMCP("Sovereign Memory")

# Conecta ao Backend (Docker)
# Certifique-se que o docker-compose up -d está rodando
import os
SOVEREIGN_KEY = os.environ.get("SOVEREIGN_KEY", "sk_sov_9988776655_COLDBREAKER_V1")
brain = SovereignBrain(base_url="http://localhost:8001", api_key=SOVEREIGN_KEY)

@mcp.tool()
def remember(fact: str, category: str = "general") -> str:
    """
    Grava uma informação importante, fato, trecho de código ou preferência na memória de longo prazo.
    Use isso quando o usuário pedir para 'lembrar' algo ou quando a informação parecer crítica para o futuro.
    
    Args:
        fact: O conteúdo exato a ser lembrado.
        category: (Opcional) Uma tag para organizar (ex: 'code', 'personal', 'work').
    """
    logger.info(f"Gravando memória: {category} | {fact[:20]}...")
    try:
        # Injetamos a categoria no texto para ajudar na busca semântica futura
        full_text = f"[{category.upper()}] {fact}"
        result = brain.learn(full_text, session_id="mcp-desktop-session")
        return f"Memória gravada com sucesso: {result}"
    except Exception as e:
        return f"Erro ao gravar memória: {str(e)}"

@mcp.tool()
def recall(query: str) -> str:
    """
    Busca na memória de longo prazo por informações relevantes baseadas em uma pergunta.
    Use isso antes de responder perguntas que possam depender de contexto histórico, projetos passados ou preferências do usuário.
    
    Args:
        query: A pergunta ou tópico para pesquisar na memória.
    """
    logger.info(f"Buscando memória: {query}")
    try:
        # Usamos o 'ask' do SDK, que já faz o RAG + Injection
        # O modelo 'memory-only' aqui seria inútil, precisamos que o Brain processe e nos dê a resposta sintetizada
        # Ou, podemos pedir apenas os fragmentos crus. Vamos pedir a síntese para ser mais útil.
        response = brain.ask(query, session_id="mcp-desktop-session")
        return response
    except Exception as e:
        return f"Erro ao consultar memória: {str(e)}"

@mcp.tool()
def inspect_memories(limit: int = 5) -> str:
    """
    Lista as memórias mais recentes com seus IDs numéricos.
    USE ISSO ANTES de tentar deletar ou editar algo, para descobrir o ID correto.
    """
    # brain.list_recent returns a list of dicts
    items = brain.list_recent(limit=limit)
    if not items:
        return "Nenhuma memória encontrada."
    
    # Check for error reporting from SDK
    if len(items) == 1 and "error" in items[0]:
         return items[0]["error"]

    report = "MEMÓRIAS RECENTES:\n"
    for item in items:
        # Formato: [ID: 123] (User): O texto...
        # Ensure we access keys safely if schema changes, though backend is controlled.
        mid = item.get('id', 'N/A')
        role = item.get('role', 'unknown')
        content = item.get('content', '')
        report += f"[ID: {mid}] ({role}): {content}\n"
    return report

@mcp.tool()
def forget_memory(memory_id: int) -> str:
    """
    Apaga permanentemente uma memória pelo seu ID numérico.
    Você deve ter usado 'inspect_memories' antes para ter certeza do ID.
    Args:
        memory_id: O número do ID para deletar (ex: 42).
    """
    return brain.forget(memory_id)

@mcp.tool()
def correct_memory(memory_id: int, new_fact: str) -> str:
    """
    Corrige/Edita uma memória existente.
    Args:
        memory_id: O ID numérico da memória.
        new_fact: O texto correto que deve substituir o antigo.
    """
    return brain.update(memory_id, new_fact)

if __name__ == "__main__":
    # O MCP roda via stdio (entrada/saída padrão) para se comunicar com o Claude Desktop
    mcp.run()

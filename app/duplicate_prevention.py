"""
Aethera Cortex Duplicate Prevention Module.
Semantic similarity check before memory persistence to prevent duplicate entries.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger("Aethera.DuplicatePrevention")

# Default similarity threshold (0.95 = 95% similar)
DEFAULT_SIMILARITY_THRESHOLD = 0.95


class DuplicateCheckResult:
    """Result of a duplicate check operation."""

    def __init__(self, is_duplicate: bool, existing_id: Optional[int] = None,
                 existing_content: Optional[str] = None, similarity: float = 0.0):
        self.is_duplicate = is_duplicate
        self.existing_id = existing_id
        self.existing_content = existing_content
        self.similarity = similarity

    def to_dict(self) -> dict:
        return {
            "is_duplicate": self.is_duplicate,
            "existing_id": self.existing_id,
            "existing_content": self.existing_content,
            "similarity": round(self.similarity, 4)
        }


def check_duplicate(embedding: list, owner_key: str,
                    get_db_connection_func, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
                    ) -> DuplicateCheckResult:
    """
    Verifica se já existe uma memória semanticamente similar no banco.

    Args:
        embedding: Vetor de embedding já calculado (384-dim)
        owner_key: Chave do dono (multi-tenancy filter)
        get_db_connection_func: Função para obter conexão com DB
        threshold: Limiar de similaridade (default: 0.95)

    Returns:
        DuplicateCheckResult com is_duplicate, existing_id, similarity

    Algorithm:
        1. Busca a memória mais similar via pgvector cosine distance
        2. Filtra por owner_key (multi-tenancy)
        3. Converte distance para similarity (1 - distance)
        4. Retorna duplicate=True se similarity >= threshold

    Side Effects:
        - Abre e fecha conexão com PostgreSQL
        - Log INFO se duplicata encontrada

    Performance:
        - Com índice HNSW: ~5-15ms
        - Sem índice: ~2-50ms dependendo do volume
    """
    try:
        conn = get_db_connection_func()
        c = conn.cursor()

        c.execute("""
            SELECT id, content, (embedding <=> %s::vector) as distance
            FROM memories
            WHERE owner_key = %s
            ORDER BY distance ASC
            LIMIT 1
        """, (embedding, owner_key))

        result = c.fetchone()
        conn.close()

        if not result:
            return DuplicateCheckResult(is_duplicate=False)

        existing_id, existing_content, distance = result
        similarity = 1.0 - distance

        if similarity >= threshold:
            logger.info(
                f"[DUPLICATE] Detected: similarity={similarity:.4f} "
                f"(threshold={threshold}) | memory_id={existing_id} | "
                f"owner={owner_key[:20]}..."
            )
            return DuplicateCheckResult(
                is_duplicate=True,
                existing_id=existing_id,
                existing_content=existing_content,
                similarity=similarity
            )

        return DuplicateCheckResult(is_duplicate=False, similarity=similarity)

    except Exception as e:
        logger.error(f"[DUPLICATE] Check failed: {e}")
        # Fail-open: em caso de erro, permitir gravação
        return DuplicateCheckResult(is_duplicate=False)


def merge_memory(existing_id: int, owner_key: str, get_db_connection_func) -> bool:
    """
    Atualiza o timestamp de uma memória existente (merge strategy).

    Em vez de criar duplicata, atualiza o timestamp da memória existente
    para refletir que o usuário referenciou essa informação novamente.

    Args:
        existing_id: ID da memória existente
        owner_key: Chave do dono (verificação de ownership)
        get_db_connection_func: Função para obter conexão com DB

    Returns:
        True se merge foi realizado, False caso contrário
    """
    try:
        conn = get_db_connection_func()
        c = conn.cursor()
        c.execute(
            "UPDATE memories SET timestamp = %s WHERE id = %s AND owner_key = %s",
            (time.time(), existing_id, owner_key)
        )
        updated = c.rowcount
        conn.commit()
        conn.close()

        if updated > 0:
            logger.info(f"[DUPLICATE] Merged: memory_id={existing_id} timestamp updated")
            return True
        return False

    except Exception as e:
        logger.error(f"[DUPLICATE] Merge failed: {e}")
        return False

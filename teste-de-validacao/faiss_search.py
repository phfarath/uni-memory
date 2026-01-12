#!/usr/bin/env python3
"""
Script para busca vetorial usando FAISS com medição de tempo.
Gera embeddings e realiza busca em um índice FAISS local com 100k vetores.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os


def create_or_load_index(index_path="faiss_index.bin", num_vectors=100000, embedding_dim=384):
    """
    Cria um novo índice FAISS ou carrega um existente.
    
    Args:
        index_path: Caminho para salvar/carregar o índice
        num_vectors: Número de vetores no índice
        embedding_dim: Dimensão dos embeddings (384 para all-MiniLM-L6-v2)
    
    Returns:
        Índice FAISS
    """
    if os.path.exists(index_path):
        print(f"Carregando índice existente de {index_path}...")
        index = faiss.read_index(index_path)
        print(f"Índice carregado com {index.ntotal} vetores.")
    else:
        print(f"Criando novo índice FAISS com {num_vectors} vetores...")
        # Criar índice L2 (distância euclidiana)
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Gerar vetores aleatórios para simular 100k documentos
        print("Gerando vetores aleatórios...")
        vectors = np.random.random((num_vectors, embedding_dim)).astype('float32')
        
        # Adicionar vetores ao índice
        index.add(vectors)
        
        # Salvar índice
        faiss.write_index(index, index_path)
        print(f"Índice criado e salvo em {index_path}")
    
    return index


def main():
    # Configurações
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_PATH = "faiss_index.bin"
    NUM_VECTORS = 100000
    TOP_K = 5  # Número de resultados a retornar
    
    # Query de exemplo
    query = "Exemplo de busca semântica em português"
    
    print("=" * 60)
    print("Busca Vetorial com FAISS")
    print("=" * 60)
    print(f"Modelo: {MODEL_NAME}")
    print(f"Índice: {NUM_VECTORS} vetores")
    print(f"Query: {query}")
    print("=" * 60)
    
    # Iniciar medição de tempo total
    start_total = time.time()
    
    # ===== PASSO 1: Carregar modelo =====
    print("\n[1/4] Carregando modelo de embeddings...")
    start_model = time.time()
    model = SentenceTransformer(MODEL_NAME)
    model_load_time = time.time() - start_model
    print(f"Modelo carregado em {model_load_time:.4f} segundos")
    
    # ===== PASSO 2: Carregar/criar índice FAISS =====
    print("\n[2/4] Preparando índice FAISS...")
    start_index = time.time()
    embedding_dim = model.get_sentence_embedding_dimension()
    index = create_or_load_index(INDEX_PATH, NUM_VECTORS, embedding_dim)
    index_load_time = time.time() - start_index
    print(f"Índice preparado em {index_load_time:.4f} segundos")
    
    # ===== PASSO 3: Gerar embedding da query =====
    print("\n[3/4] Gerando embedding da query...")
    start_embedding = time.time()
    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    embedding_time = time.time() - start_embedding
    print(f"Embedding gerado em {embedding_time:.4f} segundos")
    print(f"Dimensão do embedding: {query_embedding.shape}")
    
    # ===== PASSO 4: Realizar busca =====
    print(f"\n[4/4] Realizando busca (top-{TOP_K})...")
    start_search = time.time()
    distances, indices = index.search(query_embedding, TOP_K)
    search_time = time.time() - start_search
    print(f"Busca realizada em {search_time:.4f} segundos")
    
    # Tempo total
    total_time = time.time() - start_total
    
    # ===== Resultados =====
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"\nTop {TOP_K} resultados mais próximos:")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        print(f"  {i}. Índice: {idx:6d} | Distância: {dist:.4f}")
    
    print("\n" + "=" * 60)
    print("TEMPOS DE EXECUÇÃO")
    print("=" * 60)
    print(f"Carregamento do modelo: {model_load_time:.4f} segundos")
    print(f"Preparação do índice:    {index_load_time:.4f} segundos")
    print(f"Geração do embedding:    {embedding_time:.4f} segundos")
    print(f"Busca FAISS:             {search_time:.4f} segundos")
    print("-" * 60)
    print(f"TEMPO TOTAL (Embedding + Search): {embedding_time + search_time:.4f} segundos")
    print(f"TEMPO TOTAL (incluindo setup):    {total_time:.4f} segundos")
    print("=" * 60)


if __name__ == "__main__":
    main()

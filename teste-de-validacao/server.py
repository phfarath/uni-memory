from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import uvicorn

app = FastAPI()

# Carregamento Global (Acontece 1 vez)
print("Carregando modelo...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
# Popula com lixo só pra testar busca
vectors = np.random.random((100000, 384)).astype('float32')
index.add(vectors)
print("Servidor pronto!")

@app.get("/search")
async def search(q: str):
    start = time.time()
    
    # 1. Embedding (Onde o GIL ataca)
    embedding = model.encode([q])
    
    # 2. Busca FAISS (Libera o GIL, C++ puro)
    D, I = index.search(embedding, 5)
    
    end = time.time()
    return {"latency_ms": (end - start) * 1000}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

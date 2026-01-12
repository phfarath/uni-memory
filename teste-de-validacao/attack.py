import aiohttp
import asyncio
import time
import numpy as np

URL = "http://localhost:8000/search?q=teste de performance"
CONCURRENT_REQUESTS = 50  # Simula 50 usuários ao mesmo tempo

async def fetch(session):
    async with session.get(URL) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session) for _ in range(CONCURRENT_REQUESTS)]
        
        start_global = time.time()
        results = await asyncio.gather(*tasks)
        end_global = time.time()
        
        latencies = [r['latency_ms'] for r in results]
        
        print(f"--- RESULTADOS PARA {CONCURRENT_REQUESTS} REQUISIÇÕES SIMULTÂNEAS ---")
        print(f"Tempo Total do Batch: {end_global - start_global:.4f}s")
        print(f"Latência Média (Server-side): {np.mean(latencies):.2f}ms")
        print(f"Latência Máxima (Pior caso): {np.max(latencies):.2f}ms")
        print(f"Throughput: {CONCURRENT_REQUESTS / (end_global - start_global):.2f} req/s")

if __name__ == "__main__":
    asyncio.run(main())

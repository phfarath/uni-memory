import asyncio
import aiohttp
import json

API_KEY = "sk_aethera_kJIhaMXBPt-DaHb9uPjWPQ"
BASE_URL = "http://localhost:8001"

async def simulate_client():
    print(f"Connecting to SSE: {BASE_URL}/mcp/sse...")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/mcp/sse?x-api-key={API_KEY}") as response:
            print(f"SSE Status: {response.status}")
            
            endpoint_url = None
            
            async for line in response.content:
                decoded = line.decode('utf-8').strip()
                if not decoded: continue
                print(f"<RECV> {decoded}")
                
                if decoded.startswith("data: http"):
                    endpoint_url = decoded.replace("data: ", "")
                    print(f"\n[!] Detected Endpoint: {endpoint_url}")
                    # Start POST task
                    asyncio.create_task(send_init(session, endpoint_url))

async def send_init(session, url):
    print(f"\n[>] Sending JSON-RPC Init to {url}...")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "TestScript", "version": "1.0"}
        }
    }
    
    try:
        async with session.post(url, json=payload) as resp:
            print(f"POST Status: {resp.status}")
            print(f"POST Body: {await resp.text()}")
    except Exception as e:
        print(f"POST Failed: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_client())

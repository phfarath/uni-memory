import sys
import asyncio
import aiohttp
import json
import signal

# Configuração
# Pega a URL do argumento: python3 sse_bridge.py <SSE_URL>
if len(sys.argv) < 2:
    print("Usage: python3 sse_bridge.py <SSE_URL>", file=sys.stderr)
    sys.exit(1)

SSE_URL = sys.argv[1]
ENDPOINT_URL = None
SESSION_ID = None
API_KEY = None

# Extract initial API Key from SSE_URL if present (for the first connection)
from urllib.parse import urlparse, parse_qs
parsed = urlparse(SSE_URL)
qs = parse_qs(parsed.query)
if 'x-api-key' in qs:
    API_KEY = qs['x-api-key'][0]

async def sse_reader(session):
    global ENDPOINT_URL
    print(f"Connecting to SSE: {SSE_URL}", file=sys.stderr)
    
    async with session.get(SSE_URL) as resp:
        if resp.status != 200:
            print(f"SSE Connection Failed: {resp.status} - {await resp.text()}", file=sys.stderr)
            return

        async for line in resp.content:
            line = line.decode('utf-8').strip()
            if not line: continue
            
            if line.startswith("data: "):
                data_str = line[6:] # Remove "data: "
                
                # Check for Endpoint Event
                if data_str.startswith("http"):
                    ENDPOINT_URL = data_str
                    print(f"Endpoint Discovered: {ENDPOINT_URL}", file=sys.stderr)
                    continue
                
                # Check for JSON Message
                try:
                    # Validate it's JSON
                    json.loads(data_str)
                    # Forward to Roo Code (Stdout)
                    sys.stdout.write(data_str + "\n")
                    sys.stdout.flush()
                except ValueError:
                    pass

async def stdin_reader(session):
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line: break
        
        msg = line.decode('utf-8').strip()
        if not msg: continue
        
        if not ENDPOINT_URL:
            print("WAITING FOR ENDPOINT...", file=sys.stderr)
            # Wait a bit or buffer? 
            # In MCP, the client initiates 'initialize'. 
            # We must wait for SSE to give us the endpoint first.
            while not ENDPOINT_URL:
                await asyncio.sleep(0.1)

        # POST the message
        try:
            # MCP messages are JSON
            json_msg = json.loads(msg)
            async with session.post(ENDPOINT_URL, json=json_msg) as resp:
                if resp.status >= 400:
                    print(f"POST Error {resp.status}: {await resp.text()}", file=sys.stderr)
        except Exception as e:
            print(f"Post Failed: {e}", file=sys.stderr)

async def main():
    async with aiohttp.ClientSession() as session:
        # Run SSE Reader and Stdin Reader concurrently
        await asyncio.gather(
            sse_reader(session),
            stdin_reader(session)
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

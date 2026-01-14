from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
import inspect

print("=== FastMCP Attributes ===")
try:
    mcp = FastMCP("test")
    print(dir(mcp))
    print(f"Callable? {callable(mcp)}")
except Exception as e:
    print(f"Error instantiating FastMCP: {e}")

print("\n=== SseServerTransport Attributes ===")
try:
    auth_method = lambda x: x
    # Try instantiation if possible, constructor might vary
    try:
        transport = SseServerTransport("/endpoint")
    except:
        transport = SseServerTransport("/endpoint", "session") # Guessing
    print(dir(transport))
    print(f"Connect method? {'connect' in dir(transport)}")
    
    if 'connect_sse' in dir(transport):
        print("\n--- connect_sse details ---")
        print(f"Signature: {inspect.signature(transport.connect_sse)}")
        print(f"Docstring: {transport.connect_sse.__doc__}")
except Exception as e:
    print(f"Error instantiating Transport: {e}")

print("\n=== Server Attributes ===")
print(dir(Server))

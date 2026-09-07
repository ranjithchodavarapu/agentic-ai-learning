from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MiniProjectServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

if __name__ == "__main__":
    mcp.run()
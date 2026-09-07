# LangGraph + MCP mini-project

A LangGraph agent with conditional tool routing, checkpointed memory, and one
tool exposed via a real MCP server (stdio transport) instead of a local function.

## What it demonstrates
- StateGraph with conditional edges (`tools_condition`) for agentic tool-use decisions
- Checkpointed cross-call memory via `thread_id`
- MCP integration via `langchain-mcp-adapters` — mixing a local `@tool` and an
  MCP-exposed tool in the same agent
- Observed: the model declines to call a tool when none fits the task
  (e.g. multiplication with only an `add` tool available), reasoning through
  it manually instead of misusing the wrong tool

## Files
- `mcp_server.py` — standalone MCP server exposing `add`
- `day6.py` — LangGraph client connecting to it, plus a local `search_stub` tool

## What I learned
The graph/state model clicked faster than expected given my background, but the real lesson was operational: two different providers deprecated their model names mid-week with zero warning, which forced me to actually debug instead of just following a tutorial. The most interesting moment was watching the agent decline to misuse add for a multiplication problem and reason through it manually instead — a small but real instance of the kind of tool-selection judgment I care about in my own reliability research.
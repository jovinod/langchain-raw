# LangChain & Ollama Agent Loop — Tool Calling Exploration

Two implementations of the same agentic tool-calling loop — one using LangChain's abstractions and one using raw Ollama SDK calls — to understand what happens under the hood.

## What It Does

A shopping assistant agent that can:

1. **Look up product prices** from a catalog (laptop, smartphone, headphones)
2. **Apply discount tiers** (bronze 5%, silver 12%, gold 23%) to a price

The agent loops — calling tools, feeding results back into the conversation — until it can produce a final answer. Both implementations are traced with [LangSmith](https://smith.langchain.com/).

## Implementations

### 1. LangChain Tool Calling (`1_agent_loop_langchain_tool_calling.py`)

Uses LangChain's high-level API:

- `@tool` decorator to define tools with auto-generated schemas
- `init_chat_model()` + `.bind_tools()` to wire tools to the LLM
- `ToolMessage` to feed tool results back into the conversation
- LangChain message types (`SystemMessage`, `HumanMessage`, `AIMessage`)

### 2. Raw Ollama Function Calling (`2_agent_loop_raw_functiona_calling.py`)

Uses the Ollama Python SDK directly — no LangChain abstractions:

- Tool schemas defined manually as JSON (`tools_for_llm`)
- `ollama.chat()` called directly with `tools=` parameter
- Raw dict messages (`{"role": "tool", "content": ...}`)
- `@traceable` decorators for LangSmith tracing

## Prerequisites

- [Ollama](https://ollama.com/) running locally with the `qwen3:1.7b` model pulled
- Python 3.12+
- A `.env` file with:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your-langsmith-api-key>
LANGSMITH_PROJECT=<your-project-name>
```

## Setup

```bash
# Pull the model
ollama pull qwen3:1.7b

# Install dependencies (using uv)
uv sync

# Run the LangChain version
uv run python 1_agent_loop_langchain_tool_calling.py

# Run the raw Ollama version
uv run python 2_agent_loop_raw_functiona_calling.py
```

## Example Output

```
Question: What is the price of a laptop after applying a gold discount?
============================================================
--- Iteration 1 ---
    [Tool Selected] get_product_price with args {'product': 'laptop'}
    >> Executing get_product_price(product='laptop')
    [Tool Result] 1299.99
--- Iteration 2 ---
    [Tool Selected] apply_discount with args {'price': 1299.99, 'discount_tier': 'gold'}
    >> Executing apply_discount(price=1299.99, discount_tier='gold')
    [Tool Result] 1000.99
--- Iteration 3 ---

Final Answer: The price of a laptop after applying a gold discount is $1,000.99.
```

## Dependencies

- `langchain` / `langchain-ollama` — LangChain framework + Ollama integration
- `ollama` — Ollama Python SDK (via `langchain-ollama` dependency)
- `langsmith` — Tracing and observability
- `python-dotenv` — Environment variable loading
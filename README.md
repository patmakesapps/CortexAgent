# CortexAgent

CortexAgent is a modular orchestration layer that sits between `CortexUI` and `CortexLTM`.

v1 goal:
- Route normal chat to CortexLTM
- Trigger web search when the user asks for current/external information
- Persist both user and assistant messages to CortexLTM even when tools are used

## Architecture

- `cortexagent/router/intent_router.py`
  - Heuristic intent gating (`chat` vs `web_search`)
- `cortexagent/tools/base.py`
  - Generic tool interfaces
- `cortexagent/tools/registry.py`
  - Tool registration and lookup
- `cortexagent/tools/web_search.py`
  - Web search tool with provider abstraction
- `cortexagent/services/cortexltm_client.py`
  - HTTP client for CortexLTM endpoints
- `cortexagent/services/orchestrator.py`
  - End-to-end chat orchestration
- `cortexagent/main.py`
  - FastAPI app + routes

## Setup

1. Create venv and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure env:

```powershell
copy .env.example .env
```

3. Start server:

```powershell
uvicorn cortexagent.main:app --host 0.0.0.0 --port 8010
```

## Environment

Required:
- `CORTEXLTM_API_BASE_URL` (example: `http://127.0.0.1:8000`)

Optional:
- `CORTEXLTM_API_KEY` (forwarded as `x-api-key`)
- `AGENT_TOOLS_ENABLED` (`true`/`false`, default `true`)
- `WEB_SEARCH_ENABLED` (`true`/`false`, default `true`)
- `WEB_SEARCH_PROVIDER` (comma-separated fallback chain; e.g. `duckduckgo,bing,brave`; default `duckduckgo,bing`)
- `BRAVE_SEARCH_API_KEY` (required for Brave provider)
- `WEB_SEARCH_TIMEOUT_SECONDS` (default `8`)
- `WEB_SEARCH_MAX_RESULTS` (default `5`)
- `WEB_SEARCH_RETRIES` (per-provider retry count, default `2`)
- `AGENT_ROUTER_LLM_ENABLED` (`true`/`false`, default `true`)
- `AGENT_ROUTER_LLM_MODEL` (default: `AGENT_ROUTER_LLM_MODEL` -> `GROQ_ROUTER_MODEL` -> `GROQ_CHAT_MODEL` -> `llama-3.1-8b-instant`)
- `AGENT_ROUTER_LLM_TIMEOUT_SECONDS` (default `6`)
- `GROQ_API_KEY` (enables model-based route decisions; without it router falls back to heuristics)

## Route

- `POST /v1/agent/threads/{thread_id}/chat`
  - If tool intent is detected and enabled, runs web search and returns a cited answer.
  - Otherwise forwards to CortexLTM `/v1/threads/{thread_id}/chat`.

## Notes

- The tool system is intentionally modular for future connectors (Notion, Slack, Calendar, etc.).
- Routing is model-first (system-prompt classifier) with heuristic fallback.

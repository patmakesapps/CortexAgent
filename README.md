# CortexAgent

CortexAgent is the orchestration layer between `CortexUI` and `CortexLTM`.

## Runtime Architecture

The runtime is intentionally LLM-first and split into explicit layers:

- `planner` (`cortexagent/services/planner.py`)
  - Reads full conversation context.
  - Outputs a strict structured decision:
    - `direct_response`
    - `tool_pipeline` (ordered tool steps + args)
- `executor` (`cortexagent/services/executor.py`)
  - Deterministically executes planner steps.
  - Never infers intent or mutates plan decisions.
- `tool registry` (`cortexagent/tools/registry.py`)
  - Holds tool schema + metadata.
  - Validates planner args before execution.
- `orchestrator` (`cortexagent/services/orchestrator.py`)
  - Coordinates planner + executor + persistence.
- `api` (`cortexagent/main.py`)
  - FastAPI routes consumed by CortexUI.

No regex-based routing or keyword intent triggers are used for planner decisions.

## UI Contract

The agent response keeps legacy fields expected by CortexUI:

- `decision`
- `tool_pipeline`
- `sources`

CortexUI still receives:

- `x-cortex-agent-trace`
- `x-cortex-route-mode`

through the existing proxy behavior in `CortexUI`.

## Google Integrations

Google account connect/disconnect/status routes are provided:

- `POST /v1/agent/integrations/google/connect`
- `GET /v1/agent/integrations/google/status`
- `POST /v1/agent/integrations/google/disconnect`

Google tool adapters retained:

- Calendar: `cortexagent/tools/google_calendar.py`
- Drive: `cortexagent/tools/google_drive.py`
- Gmail: `cortexagent/tools/google_gmail.py`

## API Route

- `POST /v1/agent/threads/{thread_id}/chat`

Response model:

```json
{
  "thread_id": "string",
  "response": "assistant text",
  "decision": {
    "action": "chat|orchestration|<tool_name>",
    "reason": "planner rationale",
    "confidence": 0.0
  },
  "sources": [{"title": "string", "url": "string"}],
  "tool_pipeline": []
}
```

## Environment

Required:

- `CORTEXLTM_API_BASE_URL`
- `AGENT_PLANNER_LLM_PROVIDER`
- `AGENT_PLANNER_LLM_MODEL`
- `AGENT_PLANNER_LLM_API_KEY` (or `GROQ_API_KEY`)

Optional:

- `CORTEXLTM_API_KEY`
- `AGENT_PLANNER_LLM_TIMEOUT_SECONDS` (default `8`)
- `AGENT_PLANNER_MAX_STEPS` (default `4`)
- `AGENT_PLANNER_CONTEXT_MESSAGES` (default `10`)
- `AGENT_PLANNER_LLM_API_BASE_URL`
- `AGENT_SYNTHESIS_LLM_ENABLED` (default `true`)
- `AGENT_SYNTHESIS_LLM_PROVIDER`
- `AGENT_SYNTHESIS_LLM_MODEL`
- `AGENT_SYNTHESIS_LLM_TIMEOUT_SECONDS` (default `10`)
- `AGENT_SYNTHESIS_LLM_API_BASE_URL`
- `AGENT_SYNTHESIS_LLM_API_KEY`

Connected account / OAuth:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `CONNECTED_ACCOUNTS_TABLE` (default `ltm_connected_accounts`)
- `CONNECTED_ACCOUNTS_TIMEOUT_SECONDS` (default `8`)
- `CONNECTED_ACCOUNTS_TOKEN_ENCRYPTION_KEY` (recommended)
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `GOOGLE_OAUTH_TIMEOUT_SECONDS` (default `8`)

## Run

```powershell
uvicorn cortexagent.main:app --host 0.0.0.0 --port 8010
```

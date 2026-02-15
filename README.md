# CortexAgent

CortexAgent is a modular orchestration layer that sits between `CortexUI` and `CortexLTM`.

v1 goal:
- Route normal chat to CortexLTM
- Trigger web search when the user asks for current/external information
- Persist both user and assistant messages to CortexLTM even when tools are used
- Support Google Calendar read + write flows with explicit user confirmation
- Support Google Drive read flows for file discovery/open links
- Support Gmail list/read/draft flows, plus confirmation-gated send

## Release Notes (Feb 14, 2026)

- Improved routing precision with three explicit buckets:
  - explicit web-intent cues (for example: "search online", "look up", "show links")
  - time-sensitive fact requests (latest/news/price/schedule style questions)
  - non-web writing/coding tasks that should stay in normal chat
- Added safer verification override behavior:
  - high-stakes verification can force web search for factual question-style prompts
  - drafting/coding/translation-style prompts are not force-routed to web search
- Improved high-stakes verification responses by appending source links when checks fail.
- Added Google Calendar write support via quick-add with confirmation-first flow:
  - write intents now route to Google Calendar more reliably, including common phrasing/typos
  - draft prompts include structured event fields (title/day/time and assumptions when inferred)
  - user can confirm with `confirm` (or `confirm: ...`) and cancel with `cancel`
  - follow-up edits (for example title changes) are merged into the pending draft
- Added created-event proof links in post-write responses:
  - created entries include direct Google Calendar event links when available
- Added anti-hallucination safety for calendar writes:
  - chat-only responses are prevented from claiming an event was added unless calendar tooling executed
- Added Gmail tool support:
  - list recent inbox threads
  - read latest email or a specific `thread <id>`
  - draft replies to a thread
  - send drafts only after explicit confirmation
  - optional recipient-domain policy via `GMAIL_ALLOWED_RECIPIENT_DOMAINS`
  - prompt-injection line filtering on email content previews
- Added Google Drive tool support:
  - list/search recent files and return direct Drive links
  - routes Drive-specific prompts to `google_drive`
  - uses read-only metadata scope for safety

## Architecture

- `cortexagent/router/intent_router.py`
  - Heuristic/model intent gating (`chat` vs `web_search` vs `google_calendar` vs `google_drive` vs `google_gmail`)
- `cortexagent/tools/base.py`
  - Generic tool interfaces
- `cortexagent/tools/registry.py`
  - Tool registration and lookup
- `cortexagent/tools/web_search.py`
  - Web search tool with provider abstraction
- `cortexagent/tools/google_calendar.py`
  - Google Calendar list + create logic (confirmation-gated writes)
- `cortexagent/tools/google_gmail.py`
  - Gmail thread list/read, draft-reply, and confirmation-gated send
- `cortexagent/tools/google_drive.py`
  - Google Drive file discovery/listing and deep links (read-only)
- `cortexagent/services/cortexltm_client.py`
  - HTTP client for CortexLTM endpoints
- `cortexagent/services/orchestrator.py`
  - End-to-end chat orchestration
- `cortexagent/main.py`
  - FastAPI app + routes

## Setup

When using the top-level `Cortex.cmd` runner, CortexAgent and CortexLTM share the same
Python interpreter at `CortexLTM/.venv/Scripts/python.exe`.

1. Shared venv workflow (recommended with `Cortex.cmd`):

```powershell
cd ..\CortexLTM
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..\CortexAgent
pip install -r requirements.txt
```

2. Optional standalone venv workflow (manual CortexAgent-only runs):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Configure env:

```powershell
copy .env.example .env
```

4. Start server:

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
- `SUPABASE_URL` (required for integration connect routes that validate bearer auth)
- `SUPABASE_ANON_KEY` (required for integration connect routes that validate bearer auth)
- `SUPABASE_SERVICE_ROLE_KEY` (required for connected-account writes via Supabase PostgREST)
- `CONNECTED_ACCOUNTS_TABLE` (default `ltm_connected_accounts`)
- `CONNECTED_ACCOUNTS_TIMEOUT_SECONDS` (default `8`)
- `GOOGLE_CLIENT_ID` (required for Google OAuth code exchange)
- `GOOGLE_CLIENT_SECRET` (required for Google OAuth code exchange)
- `GOOGLE_REDIRECT_URI` (must exactly match Google OAuth client redirect URI)
- `GOOGLE_OAUTH_TIMEOUT_SECONDS` (default `8`)
- `GMAIL_ALLOWED_RECIPIENT_DOMAINS` (optional comma-separated allowlist for send safety, e.g. `gmail.com,company.com`)

## Route

- `POST /v1/agent/threads/{thread_id}/chat`
  - If tool intent is detected and enabled, runs web search, Google Calendar, Google Drive, or Gmail action.
  - Otherwise forwards to CortexLTM `/v1/threads/{thread_id}/chat`.
- `POST /v1/agent/integrations/google/connect`
  - Exchanges Google OAuth `code` for tokens, validates caller via bearer token, and upserts a row in `ltm_connected_accounts`.

## Notes

- The tool system is intentionally modular for future connectors (Notion, Slack, Calendar, etc.).
- Routing is model-first (system-prompt classifier) with heuristic fallback.
- Calendar write confirmations are intentionally explicit to avoid accidental writes.

from .cortexltm_client import CortexLTMClient
from .connected_accounts_repo import (
    ConnectedAccount,
    ConnectedAccountsRepository,
    ConnectedAccountUpsert,
    ResolvedProviderToken,
)
from .google_oauth import GoogleOAuthService
from .supabase_auth import resolve_user_id_from_authorization

__all__ = [
    "CortexLTMClient",
    "ConnectedAccount",
    "ConnectedAccountsRepository",
    "ConnectedAccountUpsert",
    "ResolvedProviderToken",
    "GoogleOAuthService",
    "resolve_user_id_from_authorization",
    "AgentOrchestrator",
    "OrchestratorResult",
]


def __getattr__(name: str):
    if name in {"AgentOrchestrator", "OrchestratorResult"}:
        from .orchestrator import AgentOrchestrator, OrchestratorResult

        return {
            "AgentOrchestrator": AgentOrchestrator,
            "OrchestratorResult": OrchestratorResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

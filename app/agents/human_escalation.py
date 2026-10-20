from app.logging_config import log_event
from app.schemas import ChatResponse, EscalationReason


class HumanEscalationAgent:
    """Deterministic handoff recommendation; no ticket or human contact is created."""

    @staticmethod
    def handle(
        message: str,
        reason: EscalationReason,
        *,
        agents_used: list[str] | None = None,
        tools_used: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> ChatResponse:
        # Do not echo a potentially sensitive or unsafe message in the response or logs.
        log_event("request_escalated", escalation_reason=reason)
        return ChatResponse(
            answer="Não consegui resolver esta solicitação com segurança. "
                   "O caso deve ser encaminhado para atendimento humano. "
                   "Este encaminhamento é simulado; nenhum atendente foi acionado.",
            status="escalated", escalation_reason=reason,
            agents_used=list(dict.fromkeys([*(agents_used or []), "human_escalation"])),
            tools_used=tools_used or [], sources=sources or [],
        )

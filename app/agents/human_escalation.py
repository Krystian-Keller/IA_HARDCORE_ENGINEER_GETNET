from app.logging_config import log_event
from app.language import answer_language
from app.schemas import ChatResponse, EscalationReason


class HumanEscalationAgent:
    """Signal an interruption. The orchestrator persists the local handoff."""

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
        answer = (
            "I could not safely resolve this request. "
            "The conversation is awaiting local human review. "
            "No external support agent has been contacted."
            if answer_language(message) == "English" else
            "Não consegui resolver esta solicitação com segurança. "
            "A conversa aguarda revisão humana local. "
            "Nenhum atendente externo foi acionado."
        )
        return ChatResponse(
            answer=answer,
            status="escalated", escalation_reason=reason,
            agents_used=list(dict.fromkeys([*(agents_used or []), "human_escalation"])),
            tools_used=tools_used or [], sources=sources or [],
        )

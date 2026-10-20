import json
from typing import get_args

import pytest

from app.agents.customer_support import CustomerSupportAgent
from app.agents.human_escalation import HumanEscalationAgent
from app.agents.knowledge import KnowledgeAgent
from app.schemas import AppError, EscalationReason
from tests.conftest import SUPPORTED


@pytest.mark.parametrize("reason", get_args(EscalationReason))
def test_human_escalation_is_deterministic(reason):
    first = HumanEscalationAgent.handle("Do not echo this private content", reason)
    second = HumanEscalationAgent.handle("Different content", reason)
    assert first == second
    assert first.status == "escalated"
    assert first.escalation_reason == reason
    assert first.agents_used == ["human_escalation"]
    assert first.tools_used == []
    assert "simulado" in first.answer


def test_escalation_keeps_executed_steps():
    result = HumanEscalationAgent.handle(
        "question", "internal_error", agents_used=["router", "knowledge"],
        tools_used=["getnet_rag"], sources=["https://www.getnet.net/"],
    )
    assert result.agents_used == ["router", "knowledge", "human_escalation"]
    assert result.tools_used == ["getnet_rag"]
    assert result.sources == ["https://www.getnet.net/"]


def test_empty_rag_escalates_without_generation(settings, gemini):
    path = settings.data_dir / "rag/chunks.json"
    path.parent.mkdir()
    path.write_text(json.dumps([{"text": "Pix payments", "source": "https://www.getnet.net/"}]))
    result = KnowledgeAgent(gemini, settings).answer("zzqxxunmatched", "knowledge_getnet")
    assert result.escalation_reason == "insufficient_information"
    assert result.tools_used == ["getnet_rag"]
    gemini.generate.assert_not_called()


def test_empty_web_escalates_without_generation(settings, gemini, monkeypatch):
    monkeypatch.setattr("app.agents.knowledge.web_search", lambda *args, **kwargs: [])
    result = KnowledgeAgent(gemini, settings).answer("Weather", "knowledge_web")
    assert result.escalation_reason == "insufficient_information"
    assert result.tools_used == ["web_search"]
    gemini.generate.assert_not_called()


def test_web_failure_escalates(settings, gemini, monkeypatch):
    def failure(*args, **kwargs):
        raise AppError(502, "Service failed")

    monkeypatch.setattr("app.agents.knowledge.web_search", failure)
    result = KnowledgeAgent(gemini, settings).answer("Weather", "knowledge_web")
    assert result.escalation_reason == "internal_error"
    assert result.tools_used == ["web_search"]


def test_empty_customer_data_escalates(settings, gemini):
    (settings.data_dir / "customers.json").write_text('{"cliente1988":{}}')
    gemini.generate.return_value = '{"tools":["get_customer_activity"]}'
    result = CustomerSupportAgent(gemini, settings).answer("My sales", "cliente1988")
    assert result.escalation_reason == "insufficient_information"
    assert result.tools_used == ["get_customer_activity"]
    gemini.generate.assert_called_once()


def test_missing_data_in_one_selected_tool_escalates(settings, gemini):
    (settings.data_dir / "customers.json").write_text(
        '{"cliente1988":{"activity":{"device":{"status":"offline"}}}}'
    )
    gemini.generate.return_value = '{"tools":["get_customer_profile","get_customer_activity"]}'
    result = CustomerSupportAgent(gemini, settings).answer("My profile and device", "cliente1988")
    assert result.escalation_reason == "insufficient_information"
    assert result.tools_used == ["get_customer_profile", "get_customer_activity"]
    gemini.generate.assert_called_once()


def test_customer_data_failure_keeps_only_attempted_tools(settings, gemini):
    (settings.data_dir / "customers.json").write_text('invalid JSON')
    gemini.generate.return_value = '{"tools":["get_customer_profile","get_customer_activity"]}'
    result = CustomerSupportAgent(gemini, settings).answer("My profile and device", "cliente1988")
    assert result.escalation_reason == "internal_error"
    assert result.tools_used == ["get_customer_profile"]


def test_router_exception_escalates(client, router, gemini, monkeypatch):
    gemini.generate.return_value = SUPPORTED
    monkeypatch.setattr(router, "classify", lambda message: (_ for _ in ()).throw(RuntimeError("private")))
    body = client.post("/chat", json={"message": "Getnet", "user_id": "x"}).json()
    assert body["escalation_reason"] == "routing_failure"
    assert "private" not in body["answer"]


def test_unexpected_request_failure_is_controlled(client, router, monkeypatch):
    def failure(request):
        raise RuntimeError("private secret")

    monkeypatch.setattr(router, "chat", failure)
    response = client.post("/chat", json={"message": "Getnet", "user_id": "x"})
    assert response.status_code == 500
    assert response.json()["escalation_reason"] == "internal_error"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert "private secret" not in response.text

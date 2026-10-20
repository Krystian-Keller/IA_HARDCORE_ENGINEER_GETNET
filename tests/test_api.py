import json
from uuid import UUID

import pytest

from app.schemas import AppError, SearchResult
from tests.conftest import ACCEPTANCE_CASES, SUPPORTED


@pytest.mark.parametrize("message,route", ACCEPTANCE_CASES)
def test_acceptance_cases_through_real_agents(client, settings, gemini, monkeypatch, message, route):
    corpus = settings.data_dir / "rag/chunks.json"
    corpus.parent.mkdir()
    corpus.write_text(json.dumps([{
        "text": "Get Clássica Get Smart Pix bank account Getnet receivables advance antecipação "
                "crediário installments WhatsApp Payment Link. This is only a test document.",
        "source": "https://www.getnet.net/en",
    }]))
    monkeypatch.setattr("app.agents.knowledge.web_search", lambda *args, **kwargs: [
        SearchResult(title="Test result", snippet="Test context", url="https://example.com/test")
    ])
    responses = [SUPPORTED, json.dumps({"route": route})]
    if route == "customer_support":
        responses.append('{"tools":["get_customer_activity"]}')
        responses.append("Resposta de teste baseada no contexto.")
    else:
        responses.append(json.dumps({"sufficient_context": True,
                                     "answer": "Resposta de teste baseada no contexto."}))
    gemini.generate.side_effect = responses
    response = client.post("/chat", json={"message": message, "user_id": "cliente1988"})
    assert response.status_code == 200
    body = response.json()
    assert UUID(body["request_id"])
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert body["route"] == route
    assert body["guardrail_category"] == "supported"
    assert body["status"] == "answered"
    assert body["answer"] == "Resposta de teste baseada no contexto."
    expected_agent = "customer_support" if route == "customer_support" else "knowledge"
    assert body["agents_used"] == ["router", expected_agent]
    tool = {"knowledge_getnet": "getnet_rag", "knowledge_web": "web_search",
            "customer_support": "get_customer_activity"}[route]
    assert body["tools_used"] == [tool]
    assert body["sources"] == ({"knowledge_getnet": ["https://www.getnet.net/en"],
                                "knowledge_web": ["https://example.com/test"],
                                "customer_support": []}[route])
    generation_prompt = gemini.generate.call_args.args[0]
    if route == "customer_support":
        assert "DEMO_DECLINED" in generation_prompt
        assert "2026-10-15" in generation_prompt
    else:
        assert body["sources"][0] in generation_prompt


def test_profile_tool_is_selected(client, gemini):
    gemini.generate.side_effect = [SUPPORTED, '{"route":"customer_support"}',
                                   '{"tools":["get_customer_profile"]}', "Dados fictícios do perfil."]
    response = client.post("/chat", json={"message": "Qual minha configuração de recebimento?",
                                         "user_id": "cliente1988"})
    assert response.status_code == 200
    assert response.json()["tools_used"] == ["get_customer_profile"]
    assert "receiving_configuration" in gemini.generate.call_args.args[0]


def test_unknown_customer_is_explicit(client, gemini):
    gemini.generate.side_effect = [SUPPORTED, '{"route":"customer_support"}',
                                   '{"tools":["get_customer_activity"]}']
    response = client.post("/chat", json={"message": "Minhas vendas", "user_id": "unknown"})
    assert response.status_code == 404
    assert "Não há dados" in response.json()["detail"]
    assert UUID(response.json()["request_id"])
    assert gemini.generate.call_count == 3


def test_missing_rag_does_not_break_health_or_web(client, gemini, monkeypatch):
    gemini.generate.side_effect = [SUPPORTED, '{"route":"knowledge_getnet"}']
    response = client.post("/chat", json={"message": "Getnet?", "user_id": "cliente1988"})
    assert response.status_code == 200
    assert response.json()["escalation_reason"] == "internal_error"
    assert client.get("/health").json() == {"status": "ok"}
    monkeypatch.setattr("app.agents.knowledge.web_search", lambda *args, **kwargs: [])
    gemini.generate.reset_mock()
    gemini.generate.side_effect = [SUPPORTED, '{"route":"knowledge_web"}']
    response = client.post("/chat", json={"message": "Clima?", "user_id": "cliente1988"})
    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["escalation_reason"] == "insufficient_information"
    assert gemini.generate.call_count == 2


@pytest.mark.parametrize("decision", ['{"route":"invalid"}', 'not json'])
def test_invalid_router_escalates(client, gemini, decision):
    gemini.generate.side_effect = [SUPPORTED, decision]
    response = client.post("/chat", json={"message": "Hello", "user_id": "x"})
    assert response.status_code == 200
    assert response.json()["escalation_reason"] == "routing_failure"
    assert response.json()["agents_used"] == ["router", "human_escalation"]


@pytest.mark.parametrize("selection", ['{"tools":[]}', '{"tools":["arbitrary_tool"]}', "not json"])
def test_invalid_tool_selection_is_controlled(client, gemini, selection):
    gemini.generate.side_effect = [SUPPORTED, '{"route":"customer_support"}', selection]
    response = client.post("/chat", json={"message": "My sales", "user_id": "x"})
    assert response.status_code == 200
    assert response.json()["escalation_reason"] == "unsupported_request"
    assert response.json()["tools_used"] == []


@pytest.mark.parametrize("payload", [
    {}, {"message": "", "user_id": "x"}, {"message": "   ", "user_id": "x"},
    {"message": "Hello", "user_id": ""}, {"message": "a" * 5001, "user_id": "x"},
])
def test_request_validation(client, gemini, payload):
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
    assert UUID(response.json()["request_id"])
    gemini.generate.assert_not_called()


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_guardrail_dependency_failure_escalates(client, gemini, status):
    gemini.generate.side_effect = AppError(status, "Serviço indisponível")
    response = client.post("/chat", json={"message": "Hello", "user_id": "x"})
    assert response.status_code == 200
    assert response.json()["status"] == "escalated"
    assert response.json()["escalation_reason"] == "internal_error"
    assert response.json()["agents_used"] == ["human_escalation"]
    assert UUID(response.json()["request_id"])


@pytest.mark.parametrize("generated,reason", [
    ('{"sufficient_context":false,"answer":"There is not enough information."}', "insufficient_information"),
    ("There is not enough information.", "internal_error"),
    ('{"sufficient_context":true,"answer":" "}', "internal_error"),
])
def test_nonempty_retrieval_does_not_make_abstention_an_answer(client, gemini, settings, generated, reason):
    corpus = settings.data_dir / "rag/chunks.json"
    corpus.parent.mkdir()
    source = "https://site.getnet.com.br/pix/"
    corpus.write_text(json.dumps([{"text": "Pix payments are available.", "source": source}]))
    gemini.generate.side_effect = [SUPPORTED, '{"route":"knowledge_getnet"}', generated]
    response = client.post("/chat", json={"message": "Do I need a bank account for Pix?", "user_id": "x"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "escalated"
    assert body["escalation_reason"] == reason
    assert body["sources"] == [source]
    assert body["agents_used"] == ["router", "knowledge", "human_escalation"]
    assert body["tools_used"] == ["getnet_rag"]

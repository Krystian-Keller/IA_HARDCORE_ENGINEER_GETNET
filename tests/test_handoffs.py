import json
from uuid import uuid4

import pytest

from app.agents.router import RouterAgent
from app.main import get_router, app
from app.schemas import AppError
from tests.conftest import SUPPORTED


def interrupt(client, gemini):
    gemini.generate.return_value = SUPPORTED
    response = client.post("/chat", json={"message":"Quero atendimento humano.", "user_id":"cliente1988"})
    assert response.status_code == 200
    return response.json()


def test_pending_state_survives_restart_and_resume_continues_conversation(client, gemini, router):
    body = interrupt(client, gemini)
    assert body["status"] == "awaiting_human" and body["requires_human"]
    restarted = RouterAgent(gemini, router.knowledge, router.support)
    app.dependency_overrides[get_router] = lambda: restarted
    with restarted.store.connection() as db:
        row = db.execute("SELECT * FROM handoffs WHERE id = ?", (body["handoff_id"],)).fetchone()
        assert row["status"] == "pending" and row["user_id"] == "cliente1988"
        assert json.loads(row["state"])["phase"] == "human_requested"
    gemini.generate.return_value = "A revisão orientou verificar a rede Wi-Fi."
    reply = client.post(f"/handoffs/{body['handoff_id']}/reply", json={"response":"Verificar a rede Wi-Fi."})
    assert reply.status_code == 200
    assert reply.json()["conversation_id"] == body["conversation_id"]
    assert reply.json()["status"] == "answered" and not reply.json()["requires_human"]
    assert "human_resume" in reply.json()["agents_used"]
    with restarted.store.connection() as db:
        row = db.execute("SELECT * FROM handoffs WHERE id = ?", (body["handoff_id"],)).fetchone()
        assert row["status"] == "resolved" and row["human_response"] == "Verificar a rede Wi-Fi."
    gemini.generate.reset_mock()
    gemini.generate.side_effect = [SUPPORTED, '{"agents":["customer_support"]}',
                                  '{"tools":["get_customer_activity"]}', "Dados fictícios da Smart."]
    followup = client.post("/chat", json={"message":"Qual a minha maquininha?", "user_id":"cliente1988",
                                          "conversation_id": body["conversation_id"]})
    assert followup.json()["status"] == "answered"
    assert "Verificar a rede Wi-Fi." in gemini.generate.call_args.args[0]


def test_pending_conversation_cannot_bypass_interrupt(client, gemini):
    body = interrupt(client, gemini)
    gemini.generate.reset_mock()
    response = client.post("/chat", json={"message":"Continue", "user_id":"cliente1988",
                                         "conversation_id": body["conversation_id"]})
    assert response.status_code == 409
    gemini.generate.assert_not_called()


@pytest.mark.parametrize("mode", ["unknown", "foreign", "resolved"])
def test_invalid_foreign_and_repeated_handoff(client, gemini, mode):
    body = interrupt(client, gemini)
    handoff_id = body["handoff_id"]
    payload = {"response":"Human note"}
    if mode == "unknown":
        handoff_id = str(uuid4())
    elif mode == "foreign":
        payload["user_id"] = "another"
    else:
        gemini.generate.return_value = "Reviewed answer."
        assert client.post(f"/handoffs/{handoff_id}/reply", json=payload).status_code == 200
    gemini.generate.reset_mock()
    response = client.post(f"/handoffs/{handoff_id}/reply", json=payload)
    assert response.status_code == (409 if mode == "resolved" else 404)
    gemini.generate.assert_not_called()


def test_failed_resume_preserves_note_and_allows_retry(client, gemini, router):
    body = interrupt(client, gemini)
    url = f"/handoffs/{body['handoff_id']}/reply"
    payload = {"response":"Check the local network."}
    gemini.generate.side_effect = AppError(503, "Provider unavailable")
    assert client.post(url, json=payload).status_code == 503
    with router.store.connection() as db:
        row = db.execute("SELECT * FROM handoffs WHERE id = ?", (body["handoff_id"],)).fetchone()
        assert row["status"] == "pending" and row["human_response"] == payload["response"]
    gemini.generate.side_effect = None
    gemini.generate.return_value = "Check the local network, as reviewed."
    assert client.post(url, json=payload).json()["status"] == "answered"
    history = router.store.history(body["conversation_id"], "cliente1988").messages
    assert sum(m.role == "human" for m in history) == 1

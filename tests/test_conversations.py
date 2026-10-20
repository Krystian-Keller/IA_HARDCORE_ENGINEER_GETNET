import json
from uuid import UUID, uuid4
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.agents.router import RouterAgent
from app.conversations import ConversationStore
from app.schemas import AppError, ChatRequest
from tests.conftest import SUPPORTED


def support_responses(answer="Dados fictícios: a Get Smart está offline no Wi-Fi."):
    return [SUPPORTED, '{"agents":["customer_support"]}',
            '{"tools":["get_customer_activity"]}', answer]


def test_create_continue_and_restart_passes_real_history(client, gemini, router):
    first_message = "Minha maquininha não conecta na internet."
    gemini.generate.side_effect = support_responses()
    first = client.post("/chat", json={"message": first_message, "user_id": "cliente1988"}).json()
    UUID(first["conversation_id"])
    assert first["status"] == "answered"
    restarted = RouterAgent(gemini, router.knowledge, router.support)
    assert [m.role for m in restarted.store.history(first["conversation_id"], "cliente1988").messages] == [
        "user", "assistant"]
    gemini.generate.reset_mock()
    gemini.generate.side_effect = support_responses("Dados fictícios da Smart: conexão Wi-Fi indisponível.")
    result = restarted.chat(ChatRequest(message="É a Smart.", user_id="cliente1988",
                                       conversation_id=first["conversation_id"]))
    assert str(result.conversation_id) == first["conversation_id"]
    assert result.status == "answered"
    # Guardrail, structured plan, tool selection and answer all receive prior turns.
    assert len(gemini.generate.call_args_list) == 4
    for call in gemini.generate.call_args_list:
        assert first_message in call.args[0]
        assert "É a Smart." in call.args[0]
    assert len(restarted.store.history(first["conversation_id"], "cliente1988").messages) == 4


def test_new_conversations_and_users_are_isolated(client, gemini):
    gemini.generate.side_effect = support_responses("PRIVATE-FIRST-CONVERSATION")
    first = client.post("/chat", json={"message": "PRIVATE-FIRST-QUESTION", "user_id": "cliente1988"}).json()
    gemini.generate.reset_mock()
    gemini.generate.side_effect = support_responses()
    second = client.post("/chat", json={"message": "My device", "user_id": "cliente1988"}).json()
    assert first["conversation_id"] != second["conversation_id"]
    assert all("PRIVATE-FIRST" not in call.args[0] for call in gemini.generate.call_args_list)
    gemini.generate.reset_mock()
    gemini.generate.return_value = '{"category":"sensitive","reason":"blocked"}'
    client.post("/chat", json={"message": "Other user", "user_id": "other"})
    assert "PRIVATE-FIRST" not in gemini.generate.call_args.args[0]


@pytest.mark.parametrize("unknown", [False, True])
def test_foreign_or_unknown_conversation_rejected_before_llm(client, gemini, unknown):
    gemini.generate.side_effect = support_responses()
    first = client.post("/chat", json={"message": "My device", "user_id": "cliente1988"}).json()
    gemini.generate.reset_mock()
    response = client.post("/chat", json={"message": "Read history", "user_id": "attacker",
        "conversation_id": str(uuid4()) if unknown else first["conversation_id"]})
    assert response.status_code == 404
    gemini.generate.assert_not_called()


def test_history_bounded_by_message_count_and_characters(tmp_path):
    store = ConversationStore(tmp_path / "test.db", max_messages=3, max_chars=100)
    request = ChatRequest(message="initial", user_id="a")
    cid, claim, _ = store.begin(request)
    store.release(cid, claim)
    with store.connection() as db:
        for index in range(10):
            db.execute("INSERT INTO messages(conversation_id, role, content) VALUES (?, 'user', ?)",
                       (cid, f"{index}:" + "x" * 50))
    history = store.history(cid, "a").messages
    assert len(history) <= 3
    assert sum(len(m.content) for m in history) == 100
    assert history[-1].content.startswith("9:")
    with pytest.raises(AppError):
        store.history(cid, "b")


def test_overlapping_turn_is_rejected_but_expired_claim_can_recover(tmp_path):
    store = ConversationStore(tmp_path / "test.db")
    cid, claim, _ = store.begin(ChatRequest(message="one", user_id="a"))
    request = ChatRequest(message="two", user_id="a", conversation_id=cid)
    def attempt():
        with pytest.raises(AppError) as error:
            store.begin(request)
        assert error.value.status_code == 409
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: attempt(), range(2)))
    with store.connection() as db:
        db.execute("UPDATE conversations SET busy_until = 0 WHERE id = ?", (cid,))
    _, replacement_claim, _ = store.begin(request)
    assert replacement_claim != claim
    store.release(cid, claim)
    with pytest.raises(AppError):
        store.begin(request)
    store.release(cid, replacement_claim)


def test_knowledge_followup_uses_user_history_for_retrieval(client, gemini, settings, monkeypatch):
    from app.rag.retriever import Retriever
    corpus = settings.data_dir / "rag/chunks.json"
    corpus.parent.mkdir()
    corpus.write_text(json.dumps([{"text": "Get Smart Wi-Fi internet connection.",
                                  "source": "https://site.getnet.com.br/maquininha/get-smart/"}]))
    queries = []
    original = Retriever.retrieve
    def capture(self, query, **kwargs):
        queries.append(query)
        return original(self, query, **kwargs)
    monkeypatch.setattr(Retriever, "retrieve", capture)
    for message, cid in [("Como funciona a conexão da Get Smart?", None), ("E o Wi-Fi?", "reuse")]:
        gemini.generate.side_effect = [SUPPORTED, '{"agents":["knowledge_getnet"]}',
                                       '{"sufficient_context":true,"answer":"A Smart usa Wi-Fi."}']
        body = {"message": message, "user_id": "cliente1988"}
        if cid:
            body["conversation_id"] = first["conversation_id"]
        first = client.post("/chat", json=body).json()
    assert "Get Smart" in queries[-1] and "E o Wi-Fi?" in queries[-1]

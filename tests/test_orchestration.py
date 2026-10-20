import json

import pytest
from pydantic import ValidationError

from app.schemas import ExecutionPlan
from tests.conftest import SUPPORTED


def test_mixed_request_executes_both_specialists_and_synthesis(client, gemini, settings, router):
    corpus = settings.data_dir / "rag/chunks.json"
    corpus.parent.mkdir()
    source = "https://site.getnet.com.br/maquininha/get-smart/"
    corpus.write_text(json.dumps([{"text":"Get Smart internet Wi-Fi connection: check network settings.",
                                  "source": source}]))
    gemini.generate.side_effect = [
        SUPPORTED, '{"agents":["customer_support","knowledge_getnet"],"needs_synthesis":true}',
        '{"tools":["get_customer_activity"]}', "Fictional Smart terminal is offline on Wi-Fi.",
        json.dumps({"sufficient_context": True, "answer": f"Check Wi-Fi network settings. {source}"}),
        f"Your fictional Smart is offline. Check Wi-Fi network settings. {source}",
    ]
    response = client.post("/chat", json={"message": "My Smart is offline. Check my terminal and explain how to fix the connection.",
                                         "user_id": "cliente1988"})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "answered"
    assert body["agents_used"] == ["router", "customer_support", "knowledge", "synthesis"]
    assert body["tools_used"] == ["get_customer_activity", "getnet_rag"]
    assert source in body["sources"] and "Wi-Fi" in body["answer"]
    prompts = [call.args[0] for call in gemini.generate.call_args_list]
    assert "Fictional Smart terminal" in prompts[4]  # Earlier specialist -> knowledge.
    assert "Fictional Smart terminal" in prompts[5] and "Check Wi-Fi" in prompts[5]
    assert len(prompts) == 6
    evidence = router.store.latest_evidence(body["conversation_id"], "cliente1988")
    assert any(item.get("tool") == "get_customer_activity" for item in evidence)
    assert any(item.get("source") == source for item in evidence)
    assert "evidence" not in body


@pytest.mark.parametrize("plan", [
    {"agents": []}, {"agents": ["customer_support"] * 3},
    {"agents": ["knowledge_getnet", "knowledge_getnet"], "needs_synthesis": True},
    {"agents": ["knowledge_getnet", "customer_support"], "needs_synthesis": True},
    {"agents": ["customer_support", "knowledge_getnet"]},
    {"agents": ["arbitrary_tool"]}, {"agents": ["customer_support"], "extra": True},
])
def test_invalid_unbounded_plans_fail_before_specialists(client, gemini, plan):
    with pytest.raises(ValidationError):
        ExecutionPlan.model_validate(plan)
    gemini.generate.side_effect = [SUPPORTED, json.dumps(plan)]
    body = client.post("/chat", json={"message": "My terminal", "user_id": "cliente1988"}).json()
    assert body["status"] == "awaiting_human"
    assert body["escalation_reason"] == "routing_failure"
    assert gemini.generate.call_count == 2


def test_first_specialist_interrupt_stops_mixed_workflow(client, gemini):
    gemini.generate.side_effect = [SUPPORTED,
        '{"agents":["customer_support","knowledge_getnet"],"needs_synthesis":true}',
        '{"tools":[]}']
    body = client.post("/chat", json={"message":"Check terminal and troubleshoot", "user_id":"cliente1988"}).json()
    assert body["status"] == "awaiting_human"
    assert "knowledge" not in body["agents_used"] and "synthesis" not in body["agents_used"]
    assert gemini.generate.call_count == 3

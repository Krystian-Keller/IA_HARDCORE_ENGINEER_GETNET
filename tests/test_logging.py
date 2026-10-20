import json
import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

from app.logging_config import JsonFormatter
from tests.conftest import SUPPORTED


def test_json_logs_correlate_with_response_without_private_data(client, gemini, caplog):
    private = "SENSITIVE-CONTENT-DO-NOT-LOG"
    gemini.generate.return_value = json.dumps({"category": "sensitive", "reason": private})
    with caplog.at_level(logging.INFO, logger="ai_hardcore_engineer"):
        response = client.post("/chat", json={"message": private, "user_id": "PRIVATE-USER"})
    request_id = response.json()["request_id"]
    UUID(request_id)
    events = [json.loads(JsonFormatter().format(record)) for record in caplog.records
              if record.name == "ai_hardcore_engineer"]
    assert [event["event"] for event in events] == [
        "request_received", "guardrail_completed", "request_escalated", "request_completed"
    ]
    assert {event["request_id"] for event in events} == {request_id}
    assert private not in json.dumps(events)
    assert "PRIVATE-USER" not in json.dumps(events)


def test_normal_flow_logs_route_tool_and_retrieval(client, gemini, monkeypatch, caplog):
    gemini.generate.side_effect = [SUPPORTED, '{"route":"knowledge_web"}']
    monkeypatch.setattr("app.agents.knowledge.web_search", lambda *args, **kwargs: [])
    with caplog.at_level(logging.INFO, logger="ai_hardcore_engineer"):
        client.post("/chat", json={"message": "Weather", "user_id": "x"})
    events = [record.getMessage() for record in caplog.records if record.name == "ai_hardcore_engineer"]
    assert events.index("guardrail_completed") < events.index("route_selected")
    assert events.index("route_selected") < events.index("tool_called") < events.index("web_search")
    assert "request_escalated" in events


def test_concurrent_requests_get_distinct_ids(client, gemini):
    gemini.generate.return_value = '{"category":"sensitive","reason":"blocked"}'
    def send(number):
        response = client.post("/chat", json={"message": "Blocked", "user_id": f"user{number}"})
        assert response.headers["X-Request-ID"] == response.json()["request_id"]
        return response.json()["request_id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(send, range(8)))
    assert len(set(ids)) == 8

import json

import pytest

from app.guardrails.guardrail import check_guardrail
from app.schemas import GuardrailResult


@pytest.mark.parametrize("category", ["supported", "sensitive", "unsafe", "irrelevant"])
def test_guardrail_validates_categories(gemini, category):
    gemini.generate.return_value = json.dumps({"category": category, "reason": "test reason"})
    result = check_guardrail("A question", gemini, user_id="cliente1988")
    assert result.category == category
    assert gemini.generate.call_args.kwargs["schema"] is GuardrailResult
    assert "cliente1988" in gemini.generate.call_args.args[0]


@pytest.mark.parametrize("raw", ["not json", "{}", '{"category":"unknown","reason":"x"}',
                                      '{"category":"supported","reason":""}',
                                      '{"category":"supported","reason":"x","extra":true}'])
def test_invalid_guardrail_fails_closed(gemini, raw):
    gemini.generate.return_value = raw
    assert check_guardrail("A question", gemini).category == "sensitive"


@pytest.mark.parametrize("category", ["sensitive", "unsafe", "irrelevant"])
def test_blocked_guardrail_does_not_reach_router_or_tools(client, gemini, category):
    gemini.generate.return_value = json.dumps({"category": category, "reason": "blocked"})
    response = client.post("/chat", json={"message": "Blocked request", "user_id": "cliente1988"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "escalated"
    assert body["escalation_reason"] == f"{category}_request"
    assert body["guardrail_category"] == category
    assert body["agents_used"] == ["human_escalation"]
    assert body["tools_used"] == []
    assert "route" not in body
    gemini.generate.assert_called_once()


def test_invalid_guardrail_cannot_dispatch(client, gemini):
    gemini.generate.return_value = "invalid JSON"
    body = client.post("/chat", json={"message": "Hello", "user_id": "x"}).json()
    assert body["status"] == "escalated"
    assert body["tools_used"] == []
    gemini.generate.assert_called_once()

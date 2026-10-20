import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import TypeAdapter

from evals.run import EvaluationCase, evaluate_case, run_evaluation


def test_case_catalog_covers_required_behaviors():
    path = Path(__file__).resolve().parents[1] / "evals/cases.json"
    cases = TypeAdapter(list[EvaluationCase]).validate_json(path.read_text())
    assert len(cases) >= 10
    assert len({case.name for case in cases}) == len(cases)
    assert {case.expected_route for case in cases} >= {
        "knowledge_getnet", "knowledge_web", "customer_support", None
    }
    assert {case.expected_guardrail for case in cases} == {"supported", "sensitive", "unsafe", "irrelevant"}
    assert any(case.expected_status == "escalated" for case in cases)


def test_evaluation_reports_pass_and_failure_without_grading_text(capsys):
    cases = [EvaluationCase(name=name, message="Weather?", user_id="x",
                            expected_route="knowledge_web", expected_tools=["web_search"])
             for name in ("pass", "fail")]
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "request_id": str(uuid4()), "route": "knowledge_web" if len(calls) == 1 else "knowledge_getnet",
            "guardrail_category": "supported", "status": "answered", "tools_used": ["web_search"],
            "agents_used": ["router", "knowledge"], "answer": "Text quality is not evaluated",
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summary = run_evaluation(cases, client, "http://test")
    assert summary == {"total_cases": 2, "passed_cases": 1, "failed_cases": 1, "accuracy": 0.5}
    assert calls == [{"message": "Weather?", "user_id": "x"}] * 2
    assert "FAILED: fail" in capsys.readouterr().out


def test_evaluation_checks_guardrail_escalation():
    case = EvaluationCase(name="blocked", message="Private data", user_id="x", expected_route=None,
                          expected_guardrail="sensitive", expected_status="escalated",
                          expected_escalation_reason="sensitive_request", expected_tools=[])
    def handler(request):
        return httpx.Response(200, json={
            "request_id": str(uuid4()), "guardrail_category": "sensitive", "status": "escalated",
            "escalation_reason": "sensitive_request", "agents_used": ["human_escalation"], "tools_used": [],
        })
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert evaluate_case(case, client, "http://test")["passed"]


def test_evaluation_can_space_requests_without_retry(monkeypatch):
    delays = []
    calls = []
    monkeypatch.setattr("evals.run.sleep", delays.append)
    cases = [EvaluationCase(name=f"case{index}", message="Weather", user_id="x",
                            expected_route="knowledge_web", expected_tools=[])
             for index in range(3)]
    def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"detail": "Unavailable"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        summary = run_evaluation(cases, client, "http://test", delay=15)
    assert delays == [15, 15]
    assert len(calls) == 3
    assert summary["failed_cases"] == 3


@pytest.mark.parametrize("body", [{}, [], {"tools_used": None}, {"request_id": None}])
def test_malformed_api_output_fails_evaluation(body):
    case = EvaluationCase(name="bad", message="Weather", user_id="x",
                          expected_route="knowledge_web", expected_tools=[])
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        assert evaluate_case(case, client, "http://test")["passed"] is False


@pytest.mark.parametrize("sources,passed", [
    (["https://www.getnet.net/en/"], False),
    (["https://site.getnet.com.br/maquininha/get-classica/"], False),
    (["https://site.getnet.com.br/maquininha/get-classica/",
      "https://site.getnet.com.br/maquininha/get-smart/"], True),
])
def test_comparison_evaluation_requires_both_product_sources(sources, passed):
    case = EvaluationCase(
        name="machines", message="Compare Get Clássica and Get Smart", user_id="x",
        expected_route="knowledge_getnet", expected_tools=["getnet_rag"],
        expected_sources=["https://site.getnet.com.br/maquininha/get-classica/",
                          "https://site.getnet.com.br/maquininha/get-smart/"],
    )
    body = {"request_id": str(uuid4()), "route": "knowledge_getnet", "guardrail_category": "supported",
            "status": "answered", "agents_used": ["router", "knowledge"],
            "tools_used": ["getnet_rag"], "sources": sources}
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        assert evaluate_case(case, client, "http://test")["passed"] is passed

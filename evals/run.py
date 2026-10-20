"""Evaluate API flow and required source coverage; factual text quality is not scored."""

import argparse
import json
from pathlib import Path
from time import sleep
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.schemas import ChatRequest, EscalationReason, GuardrailCategory, Route


class EvaluationCase(ChatRequest):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    expected_route: Route | None
    expected_guardrail: GuardrailCategory = "supported"
    expected_status: str = "answered"
    expected_escalation_reason: EscalationReason | None = None
    expected_tools: list[str]
    expected_sources: list[str] = Field(default_factory=list)


def evaluate_case(case: EvaluationCase, client: httpx.Client, base_url: str) -> dict:
    expected = {
        "http_status": 200, "route": case.expected_route,
        "guardrail_category": case.expected_guardrail, "status": case.expected_status,
        "escalation_reason": case.expected_escalation_reason,
        "tools_used": sorted(case.expected_tools), "valid_request_id": True,
        "human_escalation": case.expected_status == "escalated",
        "required_sources_present": True,
    }
    try:
        response = client.post(base_url.rstrip("/") + "/chat", json={
            "message": case.message, "user_id": case.user_id,
        })
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Response must be an object")
        try:
            UUID(body.get("request_id", ""))
            valid_request_id = True
        except (ValueError, TypeError, AttributeError):
            valid_request_id = False
        actual = {
            "http_status": response.status_code, "route": body.get("route"),
            "guardrail_category": body.get("guardrail_category"), "status": body.get("status"),
            "escalation_reason": body.get("escalation_reason"),
            "tools_used": sorted(body.get("tools_used", [])),
            "valid_request_id": valid_request_id,
            "human_escalation": "human_escalation" in body.get("agents_used", []),
            "required_sources_present": set(case.expected_sources).issubset(body.get("sources", [])),
        }
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        # Do not print request bodies, credentials, answers or exception URLs.
        actual = {"error": type(exc).__name__}
    return {"name": case.name, "passed": actual == expected, "expected": expected, "actual": actual}


def run_evaluation(
    cases: list[EvaluationCase], client: httpx.Client, base_url: str, *, delay: float = 0
) -> dict:
    results = []
    for index, case in enumerate(cases):
        if index and delay > 0:
            sleep(delay)
        result = evaluate_case(case, client, base_url)
        results.append(result)
        print(f"{'PASSED' if result['passed'] else 'FAILED'}: {case.name}", flush=True)
        if not result["passed"]:
            print("expected:", json.dumps(result["expected"], ensure_ascii=False), flush=True)
            print("actual:  ", json.dumps(result["actual"], ensure_ascii=False), flush=True)
    passed = sum(result["passed"] for result in results)
    return {
        "total_cases": len(results), "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Avalia rotas, guardrails, tools e escalation da API local.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--timeout", type=float, default=150)
    parser.add_argument("--delay", type=float, default=15,
                        help="Intervalo entre casos em segundos (padrão: 15, para reduzir HTTP 429).")
    args = parser.parse_args()
    if args.timeout <= 0 or args.delay < 0:
        parser.error("timeout deve ser positivo e delay não pode ser negativo.")
    try:
        cases = TypeAdapter(list[EvaluationCase]).validate_json(args.cases.read_text(encoding="utf-8"))
        if not cases or len({case.name for case in cases}) != len(cases):
            raise ValueError("Os casos devem ser não vazios e possuir nomes únicos.")
    except (OSError, ValueError, ValidationError) as exc:
        parser.exit(2, f"Arquivo de avaliação inválido ({type(exc).__name__}).\n")
    print("AI Hardcore Engineer Evaluation", flush=True)
    with httpx.Client(timeout=args.timeout) as client:
        summary = run_evaluation(cases, client, args.base_url, delay=args.delay)
    print(f"\nTotal: {summary['total_cases']}\nPassed: {summary['passed_cases']}"
          f"\nFailed: {summary['failed_cases']}\nAccuracy: {summary['accuracy']:.2%}")
    return 1 if summary["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

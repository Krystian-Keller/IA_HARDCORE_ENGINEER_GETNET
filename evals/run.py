"""Evaluate API flow, semantic constraints and (optionally) live Gemini judging."""

import argparse
import json
import os
import sqlite3
from pathlib import Path
from time import sleep
from typing import Literal
from uuid import UUID

import httpx
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from app.schemas import AppError, ChatRequest, EscalationReason, GuardrailCategory, Route
from app.config import Settings
from app.conversations import ConversationStore
from app.services.gemini import Gemini
from evals.quality import SemanticConstraints, check_constraints, judge_answer


class EvaluationCase(ChatRequest):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    expected_route: Route | None
    expected_guardrail: GuardrailCategory = "supported"
    expected_status: Literal["answered", "escalated", "awaiting_human"] = "answered"
    expected_escalation_reason: EscalationReason | None = None
    expected_tools: list[str]
    expected_sources: list[str] = Field(default_factory=list)
    constraints: SemanticConstraints = Field(default_factory=SemanticConstraints)
    expected_agents: list[str] = Field(default_factory=list)


def evaluate_case(case: EvaluationCase, client: httpx.Client, base_url: str, *,
                  judge: Gemini | None = None, store: ConversationStore | None = None) -> dict:
    quality_details = {}
    expected = {
        "http_status": 200, "route": case.expected_route,
        "guardrail_category": case.expected_guardrail, "status": case.expected_status,
        "escalation_reason": case.expected_escalation_reason,
        "required_tools_present": True, "valid_request_id": True,
        "human_escalation": case.expected_status in {"escalated", "awaiting_human"},
        "required_sources_present": True,
        "semantic_constraints_pass": True, "required_agents_present": True,
    }
    try:
        response = client.post(base_url.rstrip("/") + "/chat", json={
            "message": case.message, "user_id": case.user_id,
        })
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Response must be an object")
        if not isinstance(body.get("answer", ""), str):
            raise ValueError("Answer must be a string")
        for field in ("tools_used", "agents_used", "sources"):
            values = body.get(field, [])
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError("Operational metadata must contain string lists")
        try:
            UUID(body.get("request_id", ""))
            valid_request_id = True
        except (ValueError, TypeError, AttributeError):
            valid_request_id = False
        actual = {
            "http_status": response.status_code, "route": body.get("route"),
            "guardrail_category": body.get("guardrail_category"), "status": body.get("status"),
            "escalation_reason": body.get("escalation_reason"),
            "required_tools_present": (set(case.expected_tools).issubset(body.get("tools_used", []))
                                       if case.expected_tools else not body.get("tools_used", [])),
            "valid_request_id": valid_request_id,
            "human_escalation": "human_escalation" in body.get("agents_used", []),
            "required_sources_present": set(case.expected_sources).issubset(body.get("sources", [])),
            "required_agents_present": set(case.expected_agents).issubset(body.get("agents_used", [])),
        }
        evidence = None
        if store is not None:
            evidence = store.latest_evidence(str(UUID(body["conversation_id"])), case.user_id)
        checks = check_constraints(body.get("answer", ""), body.get("sources", []), case.constraints, evidence)
        quality_details["constraints"] = checks
        actual["semantic_constraints_pass"] = all(checks.values())
        if judge is not None:
            expected["judge_pass"] = True
            if not evidence:
                actual["judge_pass"] = False
                actual["judge_error"] = "missing_local_evidence"
            else:
                try:
                    verdict = judge_answer(judge, case.message, body.get("answer", ""), evidence, case.constraints)
                    actual["judge_pass"] = verdict.passed
                    quality_details["judge"] = verdict.model_dump(by_alias=True)
                except Exception as exc:
                    actual["judge_pass"] = False
                    actual["judge_error"] = type(exc).__name__
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AppError, OSError, sqlite3.Error) as exc:
        # Do not print request bodies, credentials, answers or exception URLs.
        actual = {"error": type(exc).__name__}
    return {"name": case.name, "passed": actual == expected, "expected": expected, "actual": actual,
            "quality": quality_details}


def run_evaluation(
    cases: list[EvaluationCase], client: httpx.Client, base_url: str, *, delay: float = 0,
    judge: Gemini | None = None, store: ConversationStore | None = None,
) -> dict:
    results = []
    for index, case in enumerate(cases):
        if index and delay > 0:
            sleep(delay)
        result = evaluate_case(case, client, base_url, judge=judge, store=store)
        results.append(result)
        print(f"{'PASSED' if result['passed'] else 'FAILED'}: {case.name}", flush=True)
        if "judge" in result["quality"]:
            verdict = result["quality"]["judge"]
            print("judge:", json.dumps({k: verdict[k] for k in ("groundedness", "relevance", "correctness", "pass")}),
                  flush=True)
        if not result["passed"]:
            print("expected:", json.dumps(result["expected"], ensure_ascii=False), flush=True)
            print("actual:  ", json.dumps(result["actual"], ensure_ascii=False), flush=True)
            print("constraints:", json.dumps(result["quality"].get("constraints", {})), flush=True)
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
    parser.add_argument("--judge", action="store_true", help="Opt in to live Gemini semantic judging.")
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
    settings = Settings.from_env()
    judge = None
    store = None
    if args.judge:
        if not settings.gemini_api_key.get_secret_value():
            parser.exit(2, "Live judging requires GEMINI_API_KEY.\n")
        if not (settings.data_dir / "conversations.db").is_file():
            parser.exit(2, "Live judging requires the API's local conversation database.\n")
        judge = Gemini(settings)
        store = ConversationStore(settings.data_dir / "conversations.db")
        # Refusals lack retrieval evidence; judge only supported, answered cases.
        cases = [case for case in cases if case.expected_status == "answered"]
    headers = {"X-API-Key": os.getenv("EVAL_API_KEY", "")}
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        summary = run_evaluation(cases, client, args.base_url, delay=args.delay, judge=judge, store=store)
    print(f"\nTotal: {summary['total_cases']}\nPassed: {summary['passed_cases']}"
          f"\nFailed: {summary['failed_cases']}\nAccuracy: {summary['accuracy']:.2%}")
    return 1 if summary["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

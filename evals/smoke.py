"""Opt-in live HTTP smoke for memory, mixed orchestration and local HITL.

Run after ingestion: python -m evals.smoke. This calls the configured live provider.
Only operational metadata is printed; answers and customer records are omitted.
"""

import argparse
import json
import os
from uuid import UUID

import httpx

from app.config import Settings


def run_smoke(client: httpx.Client, *, reviewer_key: str = "") -> dict:
    """Run every stage and report sanitized failures without terminating the walkthrough."""
    results = []

    def record(name: str, passed: bool, **metadata):
        results.append(passed)
        print(json.dumps({"check": name, "passed": passed, **metadata}), flush=True)

    def request(method: str, path: str, **kwargs) -> tuple[int, dict]:
        try:
            response = client.request(method, path, **kwargs)
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            return response.status_code, body
        except (httpx.HTTPError, ValueError):
            # Never print exception text, headers, answers or customer records.
            return 0, {}

    def valid_id(value) -> bool:
        try:
            UUID(value)
            return True
        except (ValueError, TypeError, AttributeError):
            return False

    def chat(message: str, conversation_id=None):
        body = {"message": message, "user_id": "cliente1988"}
        if conversation_id:
            body["conversation_id"] = conversation_id
        return request("POST", "/chat", json=body)

    for path, status in (("/health", "ok"), ("/ready", "ready")):
        code, body = request("GET", path)
        record(path, code == 200 and body.get("status") == status, http_status=code)

    code, first = chat("Minha maquininha não conecta na internet.")
    cid = first.get("conversation_id")
    first_ok = code == 200 and first.get("status") == "answered" and valid_id(cid)
    record("first_turn", first_ok, http_status=code, status=first.get("status"))
    if first_ok:
        code, second = chat("É a Smart.", cid)
        record("continuation", code == 200 and second.get("status") == "answered" and
               second.get("conversation_id") == cid, http_status=code, status=second.get("status"))
    else:
        record("continuation", False, reason="first_turn_not_answered")

    code, mixed = chat("My Smart terminal is offline. Check my terminal information and tell me how to fix the connection.")
    agents = mixed.get("agents_used", [])
    record("mixed", code == 200 and mixed.get("status") == "answered" and isinstance(agents, list) and
           all(isinstance(agent, str) for agent in agents) and
           {"customer_support", "knowledge", "synthesis"}.issubset(agents),
           http_status=code, status=mixed.get("status"))

    code, pending = chat("Quero atendimento humano.")
    hid = pending.get("handoff_id")
    pending_ok = (code == 200 and pending.get("status") == "awaiting_human" and
                  pending.get("requires_human") is True and valid_id(hid) and
                  valid_id(pending.get("conversation_id")))
    record("interrupt", pending_ok, http_status=code, status=pending.get("status"))
    if pending_ok:
        code, resumed = request("POST", f"/handoffs/{hid}/reply",
            headers={"X-Reviewer-Key": reviewer_key},
            json={"response": "Revisão local: o registro fictício indica Get Smart offline no Wi-Fi. "
                               "Oriente verificar a disponibilidade da rede; nenhum ajuste remoto foi feito."})
        record("resume", code == 200 and resumed.get("status") == "answered" and
               resumed.get("requires_human") is False and
               resumed.get("conversation_id") == pending.get("conversation_id"), http_status=code)
    else:
        record("resume", False, reason="no_valid_pending_handoff")
    summary = {"passed": sum(results), "failed": len(results) - sum(results), "total": len(results)}
    print(json.dumps(summary), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.gemini_api_key.get_secret_value():
        parser.exit(2, "Live smoke requires GEMINI_API_KEY.\n")
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=240,
                      headers={"X-API-Key": os.getenv("EVAL_API_KEY", "")}) as client:
        summary = run_smoke(client, reviewer_key=settings.reviewer_api_key.get_secret_value())
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())

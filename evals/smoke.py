"""Opt-in live HTTP smoke for memory, mixed orchestration and local HITL.

Run after ingestion: python -m evals.smoke. This calls the configured live provider.
Only operational metadata is printed; answers and customer records are omitted.
"""

import argparse
import json
import os

import httpx

from app.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    settings = Settings.from_env()
    if not settings.gemini_api_key.get_secret_value():
        parser.exit(2, "Live smoke requires GEMINI_API_KEY.\n")
    results = []
    def record(name: str, passed: bool, **metadata):
        results.append(passed)
        print(json.dumps({"check": name, "passed": passed, **metadata}), flush=True)

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=240,
                      headers={"X-API-Key": os.getenv("EVAL_API_KEY", "")}) as client:
        for path in ("/health", "/ready"):
            response = client.get(path)
            record(path, response.status_code == 200, http_status=response.status_code)

        def chat(message, conversation_id=None):
            body = {"message": message, "user_id": "cliente1988"}
            if conversation_id:
                body["conversation_id"] = conversation_id
            return client.post("/chat", json=body).json()

        first = chat("Minha maquininha não conecta na internet.")
        cid = first.get("conversation_id")
        record("first_turn", first.get("status") == "answered", status=first.get("status"), conversation_id=cid)
        if cid and first.get("status") == "answered":
            second = chat("É a Smart.", cid)
            record("continuation", second.get("status") == "answered" and second.get("conversation_id") == cid,
                   status=second.get("status"))
        else:
            record("continuation", False, reason="first_turn_not_answered")
        mixed = chat("My Smart terminal is offline. Check my terminal information and tell me how to fix the connection.")
        record("mixed", mixed.get("status") == "answered" and
               {"customer_support", "knowledge", "synthesis"}.issubset(mixed.get("agents_used", [])),
               status=mixed.get("status"), agents_used=mixed.get("agents_used", []))
        pending = chat("Quero atendimento humano.")
        hid = pending.get("handoff_id")
        record("interrupt", pending.get("status") == "awaiting_human" and bool(hid), status=pending.get("status"))
        if hid:
            response = client.post(f"/handoffs/{hid}/reply",
                headers={"X-Reviewer-Key": settings.reviewer_api_key.get_secret_value()},
                json={"response": "Revisão local: o registro fictício indica Get Smart offline no Wi-Fi. "
                                   "Oriente verificar a disponibilidade da rede; nenhum ajuste remoto foi feito."})
            resumed = response.json()
            record("resume", response.status_code == 200 and resumed.get("status") == "answered" and
                   resumed.get("conversation_id") == pending.get("conversation_id"), http_status=response.status_code)
    print(json.dumps({"passed": sum(results), "failed": len(results) - sum(results), "total": len(results)}))
    return int(not all(results))


if __name__ == "__main__":
    raise SystemExit(main())

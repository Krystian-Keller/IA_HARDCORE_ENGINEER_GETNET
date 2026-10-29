# Demonstration walkthrough

Use the existing `.env`; never display its contents or credentials during the demo.
The commands below call Gemini only when explicitly noted. Readiness does not call it.
The complete ten-question acceptance table and curl/Swagger walkthrough are in
[MANUAL_TESTS.md](MANUAL_TESTS.md); requirement traceability is in [SPEC.md](SPEC.md).

1. **Architecture and setup.** Show the README diagram, `app/agents/router.py`,
   `app/conversations.py` and `app/services/gemini.py`. Plans are validated, contain at
   most two specialists and never replan. SQLite is the only new persistence component.

   ```bash
   docker compose config --quiet
   docker compose up --build -d --force-recreate
   curl -sS http://localhost:8000/health
   curl -sS http://localhost:8000/ready
   ```

2. **RAG.** Run `docker compose exec api python -m app.rag.ingest`. Explain the local
   TF-IDF/cosine pipeline and literal widget extraction. Ask in `/docs`:
   “What's the difference between the Get Clássica and the Get Smart?” Show the returned
   source URLs and inspect answer quality. A handoff is valid when evidence is insufficient.

3. **Customer tools.** Ask “What are my receiving configuration and registered business
   profile?” with `user_id=cliente1988`. Show `get_customer_profile` in `tools_used` and
   the fictional-data disclaimer. Data has a fixed 2026-10-15 snapshot; no payment runs.

4. **Multi-turn memory and restart.** Use the first two curl examples in README:
   “Minha maquininha não conecta na internet.” then “É a Smart.” with the same returned
   `conversation_id`. Between turns, optionally run `docker compose restart api` to
   demonstrate persistence. The Router and specialist receive the first turn. Reusing
   the conversation with a different `user_id` must return `404`.

5. **Mixed orchestration.** First ask “Check my terminal model and connection status,
   then compare Get Smart and Get Clássica using official product documentation.” This
   combines the fixed customer record with product facts available in the corpus.
   Also ask “My Smart terminal is offline. Check my terminal
   information and tell me how to fix the connection.” Inspect `agents_used` for
   `customer_support`, `knowledge` and `synthesis`, with customer activity and Getnet RAG
   in `tools_used`. If live pages do not support troubleshooting, show the interruption
   and the deterministic mixed-flow test as separate evidence of orchestration.

6. **Human interrupt and resume.** Ask “Quero atendimento humano.” Save `handoff_id` and
   `conversation_id`. Show `awaiting_human` and `requires_human=true`; another chat in
   that conversation returns `409`. Restart the API, then use README's reviewer curl.
   The note is saved before generation, and the final response is persisted. Repeating
   the same resolved review returns `409`. Continue chatting in the same conversation.
   No external operator receives a notification.

7. **Deterministic verification.** Run `docker compose exec api python -m pytest -q`.
   Tests mock external services and exercise the real API, SQLite, restart/retry state,
   bounded plans, specialist execution, synthesis, identity and quality validators.
   They do not establish that Gemini or web search is currently available.

8. **Explicit live checks.** With credentials/network, run:

   ```bash
   docker compose exec api python -m evals.smoke
   docker compose exec api python -m evals.run
   docker compose exec api python -m evals.run --judge
   ```

   The dataset checks flow and declared concepts/citations/language. The opt-in judge
   additionally receives actual persisted tool/retrieval evidence and returns validated
   1–5 scores. Gemini judging Gemini is not independent evaluation. Inspect failed
   checks honestly; provider 429/503 errors and missing evidence must remain failures.

9. **Identity and limitations.** Explain unauthenticated default mode versus optional
   environment-configured API-key mapping and separate reviewer credential. Show an
   identity mismatch rejection when configured. Explain plaintext local retention,
   bounded history, fixed synthetic records, provider quotas, limited retrieval,
   lack of production authentication and single-instance demonstration scope.

Finish by inspecting `docker compose logs --tail 100 api` for operational failures.
Do not claim successful live or semantic evaluation unless its actual output passed.

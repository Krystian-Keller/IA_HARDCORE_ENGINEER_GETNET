# Implementation report — 2026-10-20

## A. Architecture changes

Preserved FastAPI, Gemini, TF-IDF/cosine Getnet RAG, DuckDuckGo search, synthetic customer
tools, privacy-conscious JSON logging and Docker Compose. Added one local SQLite store,
bounded orchestration, persisted local human review, optional demo identity and answer-quality
evaluation. No dependencies or external infrastructure were added.

## B. Files created / modified

Created:

- `app/conversations.py`: conversation messages, execution claims, evidence and handoffs.
- `app/identity.py`: optional customer/reviewer credential checks.
- `app/language.py`: explicit Portuguese/English output hints.
- `evals/quality.py`: semantic constraints and validated Gemini judge.
- `evals/smoke.py`: explicitly invoked live HTTP workflow checks.
- `tests/test_conversations.py`, `tests/test_orchestration.py`, `tests/test_handoffs.py`,
  `tests/test_identity_readiness.py`, `tests/test_semantic_eval.py`.
- `DEMO.md` and this report.

Modified:

- `app/main.py`, `app/config.py`, `app/schemas.py` and `app/guardrails/guardrail.py`.
- `app/agents/router.py`, `customer_support.py`, `knowledge.py`, `human_escalation.py`.
- `evals/run.py`, `evals/cases.json` and `README.md`.
- `tests/test_api.py`, `test_escalation.py`, `test_evals.py`, `test_guardrail.py`,
  `test_services.py`: intentional handoff status changes and additional validation.
- `.env.example`, `.gitignore`, `.dockerignore`.

The real `.env`, customer dataset, requirements, ingestion/retrieval algorithms, Dockerfile
and Compose service/volume definition were not modified. No `SPEC.md` existed.

## C. Memory

`data/conversations.db` stores owned UUID conversations, user/assistant/human messages and
operational execution metadata/evidence. Prompts use at most 12 previous messages / 12,000
characters, configurable through environment variables. Router, guardrail, tool selection and
specialist generation receive that history. Missing/foreign conversations return the same
404 response. Short SQLite transactions prevent overlapping turns; crashed-worker claims
expire after ten minutes. The existing data volume survives application/container restart.

## D. Orchestration

Pydantic plans allow one specialist or support → Getnet knowledge → synthesis. Maximum two
specialist steps, no repeats, no replanning, and no later execution after an interruption.
Previous specialist results and history are passed forward. Operational metadata merges the
executed agents/tools/sources; reasoning is not exposed or stored.

## E. Human review

An interruption persists a pending handoff with ownership, original request, workflow phase,
plan/completed steps and evidence. The API returns `awaiting_human` with IDs and blocks new
turns in that conversation until resolution. The reply endpoint saves the reviewer note
before Gemini finalization, persists the final answer and marks the review resolved. Failed
generation retains the note and permits retrying it without duplicate human messages.
Unknown/foreign reviews return 404; resolved or overlapping reviews return 409. Resume
finalizes from human context rather than rerunning remaining specialists. No external human
is contacted.

## F. Semantic evaluation

The 16-case catalog preserves the original 15 cases and adds a mixed request. It supports
required concepts/tools/agents/sources, forbidden phrases, citation checks and a transparent
language heuristic. `--judge` explicitly opts into Gemini with actual persisted evidence,
question, answer and constraints. Structured 1–5 groundedness/relevance/correctness scores,
unsupported claims and verdict consistency are validated. Missing evidence and malformed
judge output fail. Gemini judging Gemini is not independent assessment. Unit tests never
call it.

## G. Security / identity

Optional environment API-key mappings enforce the authenticated synthetic identity against
`user_id`. A distinct reviewer credential is mandatory when customer authentication is
enabled. Missing/invalid keys return 401; customer identity mismatch returns 403. Default
mode remains an explicitly documented unauthenticated local demo. Credentials are hidden
from settings representations/validation error text. History/evidence are scoped by owner;
evidence is excluded from HTTP responses. Existing prompt injection boundaries and private
logging behavior are preserved. SQLite files are ignored by Git and image builds.

## H. Commands actually executed

```bash
docker compose config --quiet
docker compose up --build -d --force-recreate
docker compose exec -T api python -m app.rag.ingest
docker compose exec -T api python -m pytest -q
docker compose exec -T api python -m evals.run --delay 15
docker compose exec -T api python -m evals.run --judge --delay 15
docker compose exec -T api python -m evals.smoke
docker compose exec -T api python -m evals.run --judge --cases /tmp/getnet-semantic-cases.json --delay 15
docker compose logs --tail 100 api
python3 -m compileall -q app evals tests
git diff --check
```

Also ran intermediate suites with the repository mounted read-only in a disposable Compose
container, safe settings-presence checks, live HTTP/SQLite scripts and shorter log tails.
The two-case file under the container's `/tmp` was selected from the checked-in catalog.
The native virtual-environment Python executable was invalid (zero-byte file / permission
denied), and system Python had no pytest; final verification therefore used Docker. A host
PYTHONPATH attempt was interrupted and is not counted as a passing test run. No existing
lint/type-check configuration was found. No git push, history rewrite or real-secret edit
was performed.

## I. Deterministic test results

Final rebuilt image: **197 passed, 0 failed**, in 4.17 seconds. One third-party
Starlette/httpx deprecation warning. All normal tests used mocked provider/search/downloads;
the real API, SQLite persistence, isolation and interrupt/resume code ran in these tests.
Compilation and `git diff --check` passed.

## J. Live checks actually executed

- Compose configuration, image build and service startup succeeded.
- Real Getnet ingestion: **11 pages, 116 chunks, 0 ingestion failures**.
- `/health` and `/ready`: **HTTP 200**; Gemini connectivity honestly remained `not_checked`.
- Full live catalog run: **5 passed, 11 failed / 16**. This run preceded the explicit
  language hint refinement. Failures included provider 503 errors, answer language
  constraints and insufficient troubleshooting evidence; they were not masked as success.
- Full live judge run: **1 passed, 11 failed / 12 answered-case expectations**. Provider
  errors and missing evidence were reported as failures. A judge verdict could pass even
  while the overall case failed another constraint.
- Live workflow smoke: **6 passed, 1 failed / 7**. First chat, same-ID continuation,
  local human interruption and human resume passed. The mixed request executed support
  and knowledge but interrupted for insufficient repair evidence, so synthesis was not
  claimed as a live success. The deterministic mixed/synthesis test passed.
- After another complete container recreation, an explicit store check confirmed the four
  messages from the successful two-turn conversation and the resolved reviewer note still
  existed. HTTP continuation under a different user returned **404** before provider access.
- Focused final semantic recheck: **1 passed, 1 failed / 2**. Pix passed all declared
  constraints and received **5/5 groundedness, 5/5 relevance and 5/5 correctness** from
  Gemini. Crediário failed because Gemini returned **503 during the guardrail call**;
  no answer/evidence or successful judge result was fabricated for that case.
- Operational Docker logs showed controlled provider errors and insufficient-evidence
  handoffs; no unexpected application traceback was observed in the inspected logs.

## K. Remaining limitations and deliberate simplicity

SQLite and explicit Python control flow provide persistence and bounded workflows without
an ORM, vector database, queue or agent framework. This fits the challenge's local scope.
Stored history is plaintext with no retention policy; prompts are bounded and may forget
older details. A crashed in-flight turn may wait for claim expiry, and lost HTTP replies are
not idempotent retries. Human review has no UI, notifications or automatic timeout.
Authentication and the bilingual hints are demonstrations rather than production systems.
Public Getnet pages may not contain the requested repair procedure; the live mixed example
correctly stopped rather than inventing one. Gemini outages/quotas affect live reliability,
and semantic evaluation does not independently prove factual correctness. Production
readiness is not claimed.

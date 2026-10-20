"""SQLite conversation and handoff state. No model reasoning is stored.

Short atomic transactions claim a turn; no transaction stays open during LLM calls.
Claims expire after ten minutes so an interrupted worker cannot lock a conversation forever.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from time import time
from uuid import UUID, uuid4

from app.schemas import AppError, ChatRequest, ChatResponse, ConversationContext, Message


class ConversationStore:
    def __init__(self, path: Path, *, max_messages: int = 12, max_chars: int = 12000):
        self.path = path
        self.max_messages = max_messages
        self.max_chars = max_chars
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    claim TEXT, busy_until REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS message_history ON messages(conversation_id, id);
                CREATE TABLE IF NOT EXISTS handoffs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    user_id TEXT NOT NULL, status TEXT NOT NULL,
                    state TEXT NOT NULL, human_response TEXT, result TEXT
                );
                CREATE INDEX IF NOT EXISTS handoff_conversation ON handoffs(conversation_id, status);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def check(self) -> None:
        with self.connection() as db:
            db.execute("SELECT id FROM conversations LIMIT 1").fetchone()

    def _owned(self, db, conversation_id: str, user_id: str):
        row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        # Same response for absent and foreign conversations avoids revealing their existence.
        if row is None or row["user_id"] != user_id:
            raise AppError(404, "Conversa não encontrada para este usuário.")
        return row

    def _history(self, db, conversation_id: str) -> ConversationContext:
        rows = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (conversation_id, self.max_messages),
        ).fetchall()
        remaining = self.max_chars
        messages = []
        for row in rows:
            if remaining <= 0:
                break
            content = row["content"][-remaining:]
            messages.append(Message(role=row["role"], content=content))
            remaining -= len(content)
        return ConversationContext(messages=list(reversed(messages)))

    def history(self, conversation_id: str, user_id: str) -> ConversationContext:
        with self.connection() as db:
            self._owned(db, conversation_id, user_id)
            return self._history(db, conversation_id)

    def begin(self, request: ChatRequest) -> tuple[str, str, ConversationContext]:
        conversation_id = str(request.conversation_id or uuid4())
        claim = str(uuid4())
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if request.conversation_id is None:
                db.execute("INSERT INTO conversations(id, user_id) VALUES (?, ?)",
                           (conversation_id, request.user_id))
            row = self._owned(db, conversation_id, request.user_id)
            if row["busy_until"] > time():
                raise AppError(409, "Conversa em processamento.")
            if db.execute("SELECT id FROM handoffs WHERE conversation_id = ? AND status = 'pending'",
                          (conversation_id,)).fetchone():
                raise AppError(409, "Conversa aguardando resposta humana.")
            context = self._history(db, conversation_id)
            db.execute("UPDATE conversations SET claim = ?, busy_until = ? WHERE id = ?",
                       (claim, time() + 600, conversation_id))
            db.execute("INSERT INTO messages(conversation_id, role, content) VALUES (?, 'user', ?)",
                       (conversation_id, request.message))
        return conversation_id, claim, context

    def release(self, conversation_id: str, claim: str) -> None:
        with self.connection() as db:
            db.execute("UPDATE conversations SET claim = NULL, busy_until = 0 WHERE id = ? AND claim = ?",
                       (conversation_id, claim))

    def _check_claim(self, db, conversation_id: str, claim: str):
        row = db.execute("SELECT claim FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None or row["claim"] != claim:
            raise AppError(409, "Execução substituída; tente continuar a conversa.")

    def finish(self, conversation_id: str, user_id: str, claim: str,
               result: ChatResponse, state: dict) -> ChatResponse:
        result.conversation_id = UUID(conversation_id)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_claim(db, conversation_id, claim)
            if result.status == "escalated":
                result.status = "awaiting_human"
                result.requires_human = True
                result.handoff_id = uuid4()
                state.update(response=result.model_dump(mode="json"), evidence=result.evidence)
                db.execute("INSERT INTO handoffs(id, conversation_id, user_id, status, state) "
                           "VALUES (?, ?, ?, 'pending', ?)",
                           (str(result.handoff_id), conversation_id, user_id, json.dumps(state)))
            self._save_answer(db, conversation_id, result)
            db.execute("UPDATE conversations SET claim = NULL, busy_until = 0 WHERE id = ?", (conversation_id,))
        return result

    def _save_answer(self, db, conversation_id: str, result: ChatResponse):
        metadata = result.model_dump(mode="json", exclude={"answer"})
        metadata["evidence"] = result.evidence
        db.execute("INSERT INTO messages(conversation_id, role, content, metadata) VALUES (?, 'assistant', ?, ?)",
                   (conversation_id, result.answer, json.dumps(metadata)))

    def claim_handoff(self, handoff_id: str, user_id: str | None = None) -> tuple[dict, str, ConversationContext]:
        claim = str(uuid4())
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM handoffs WHERE id = ?", (handoff_id,)).fetchone()
            if row is None or (user_id is not None and row["user_id"] != user_id):
                raise AppError(404, "Encaminhamento não encontrado.")
            if row["status"] != "pending":
                raise AppError(409, "Encaminhamento já resolvido.")
            conv = self._owned(db, row["conversation_id"], row["user_id"])
            if conv["busy_until"] > time():
                raise AppError(409, "Conversa em processamento.")
            db.execute("UPDATE conversations SET claim = ?, busy_until = ? WHERE id = ?",
                       (claim, time() + 600, row["conversation_id"]))
            context = self._history(db, row["conversation_id"])
        return dict(row), claim, context

    def resolve(self, handoff: dict, claim: str, human_response: str, result: ChatResponse):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_claim(db, handoff["conversation_id"], claim)
            self._save_answer(db, handoff["conversation_id"], result)
            db.execute("UPDATE handoffs SET status = 'resolved', human_response = ?, result = ? WHERE id = ?",
                       (human_response, result.model_dump_json(), handoff["id"]))
            db.execute("UPDATE conversations SET claim = NULL, busy_until = 0 WHERE id = ?",
                       (handoff["conversation_id"],))

    def record_human_reply(self, handoff: dict, claim: str, response: str):
        """Save before generation; a failed resume can retry the same note after restart."""
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_claim(db, handoff["conversation_id"], claim)
            row = db.execute("SELECT human_response FROM handoffs WHERE id = ?", (handoff["id"],)).fetchone()
            if row["human_response"] is not None:
                if row["human_response"] != response:
                    raise AppError(409, "Reenvie a mesma resposta humana para retomar esta revisão.")
                return
            db.execute("UPDATE handoffs SET human_response = ? WHERE id = ?", (response, handoff["id"]))
            db.execute("INSERT INTO messages(conversation_id, role, content) VALUES (?, 'human', ?)",
                       (handoff["conversation_id"], response))

    def latest_evidence(self, conversation_id: str, user_id: str) -> list[dict]:
        with self.connection() as db:
            self._owned(db, conversation_id, user_id)
            row = db.execute("SELECT metadata FROM messages WHERE conversation_id = ? AND role = 'assistant' "
                             "ORDER BY id DESC LIMIT 1", (conversation_id,)).fetchone()
            return json.loads(row["metadata"]).get("evidence", []) if row else []

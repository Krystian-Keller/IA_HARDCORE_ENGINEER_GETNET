"""Small explicit Portuguese/English output hint; no translation service required."""

import re
import unicodedata

from app.schemas import ConversationContext


def answer_language(message: str, context: ConversationContext | None = None) -> str:
    candidates = [message]
    if context:
        candidates += [item.content for item in reversed(context.messages) if item.role == "user"]
    markers = {
        "Portuguese": {"qual", "quais", "como", "minha", "meu", "voce", "nao", "preciso",
                       "quero", "maquininha", "recebimento", "conecta", "e", "a", "o", "para"},
        "English": {"what", "when", "how", "my", "your", "you", "the", "does", "do", "can",
                    "is", "are", "and", "check", "tell", "please", "need", "with"},
    }
    for candidate in candidates:
        text = "".join(c for c in unicodedata.normalize("NFKD", candidate.casefold())
                       if not unicodedata.combining(c))
        words = set(re.findall(r"\w+", text))
        pt = len(words & markers["Portuguese"])
        en = len(words & markers["English"])
        if pt != en:
            return "Portuguese" if pt > en else "English"
    return "Portuguese"

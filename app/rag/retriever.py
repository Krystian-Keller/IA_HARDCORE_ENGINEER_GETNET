import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

from pydantic import TypeAdapter, ValidationError
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import AppError, Chunk

STOP_WORDS = list(ENGLISH_STOP_WORDS | set(
    "get getnet de da das do dos e em para por com um uma o os a as que qual quais como "
    "eu meu minha meus minhas se no na nos nas ao aos ou ser ter sao esta voce voces".split()
))
QUERY_TERMS = {
    "bank account": "conta bancaria banco conta corrente",
    "receive": "receber recebimento",
    "sales": "vendas",
    "payment link": "link de pagamento",
    "receivables advance": "antecipacao recebiveis",
    "installments": "parcelas parcelamento",
    "card machine": "maquininha maquina cartao",
    "difference": "diferenca comparar",
    "need": "preciso necessario",
}


def normalize(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def expand_query(query: str) -> str:
    normalized = normalize(query)
    additions = [terms for phrase, terms in QUERY_TERMS.items()
                 if re.search(r"\b" + re.escape(phrase) + r"\b", normalized)]
    return " ".join([query, *additions])


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise AppError(503, "A base RAG está vazia. Execute python -m app.rag.ingest.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode", ngram_range=(1, 2), stop_words=STOP_WORDS
        )
        try:
            self.matrix = self.vectorizer.fit_transform(
                f"{urlparse(str(chunk.source)).path.replace('-', ' ')} {chunk.text}" for chunk in chunks
            )
        except ValueError as exc:
            raise AppError(503, "A base RAG não possui vocabulário utilizável.") from exc

    @classmethod
    def from_file(cls, path: Path) -> "Retriever":
        try:
            chunks = TypeAdapter(list[Chunk]).validate_python(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise AppError(
                503, "Base RAG indisponível. Execute python -m app.rag.ingest."
            ) from exc
        return cls(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[Chunk]:
        if top_k < 1:
            return []
        products = list(dict.fromkeys(re.findall(
            r"\bget\s+(classica|smart|mini|lite|renova)\b", normalize(query)
        )))
        expanded = expand_query(query)
        if len(products) > 1:
            expanded += " ficha tecnica caracteristicas funcionalidades teclado visor bateria aplicativos estoque"
        scores = cosine_similarity(self.vectorizer.transform([expanded]), self.matrix)[0]
        indexes = [int(i) for i in scores.argsort()[::-1] if scores[i] > 0]
        selected = []
        # A comparison must not lose one of the named products to repeated chunks of the other.
        if len(products) > 1:
            product_indexes = []
            for product in products:
                candidates = [i for i in indexes if f"/maquininha/get-{product}/" in str(self.chunks[i].source)]
                if candidates:
                    selected.append(candidates[0])
                    product_indexes.extend(candidates)
            if len(selected) == len(products):
                indexes = [i for i in indexes if i in product_indexes]
        for i in indexes:
            if len(selected) >= top_k:
                break
            if i not in selected and sum(self.chunks[j].source == self.chunks[i].source for j in selected) < 2:
                selected.append(i)
        return [self.chunks[i] for i in selected[:top_k]]

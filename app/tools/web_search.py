from ddgs import DDGS
from pydantic import ValidationError

from app.schemas import AppError, SearchResult


def web_search(query: str, *, max_results: int = 5, timeout: float = 15) -> list[SearchResult]:
    try:
        results = DDGS(timeout=int(timeout)).text(
            query, max_results=max_results, backend="duckduckgo"
        )
        validated = []
        for item in results:
            try:
                validated.append(SearchResult(
                    title=item.get("title", ""), snippet=item.get("body", ""),
                    url=item.get("href", ""),
                ))
            except ValidationError:
                continue
        return validated[:max_results]
    except Exception as exc:
        # DDGS backends expose several engine-specific/network exceptions.
        raise AppError(502, "A busca DuckDuckGo está indisponível no momento.") from exc

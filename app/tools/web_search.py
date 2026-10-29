from ddgs import DDGS
from pydantic import ValidationError

from app.schemas import AppError, SearchResult
from app.tools.web_pages import enrich_results


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
    except Exception as exc:
        # DDGS backends expose several engine-specific/network exceptions.
        raise AppError(502, "A busca DuckDuckGo está indisponível no momento.") from exc
    return enrich_results(validated[:max_results], timeout=timeout)

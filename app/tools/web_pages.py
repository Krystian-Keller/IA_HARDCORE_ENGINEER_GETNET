"""Bounded HTML retrieval for public search results; never execute page scripts."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from ipaddress import ip_address
import socket
from time import monotonic

from bs4 import BeautifulSoup
import httpx

from app.schemas import SearchResult

MAX_PAGE_BYTES = 750_000
MAX_CONTENT_CHARS = 12_000
MAX_PAGES = 5
MAX_WORKERS = 3
MAX_REDIRECTS = 3


def public_target(url: str) -> tuple[httpx.URL, str]:
    """Resolve once, validate every address, and pin the HTTP connection to that IP."""
    parsed = httpx.URL(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.host or parsed.userinfo
            or parsed.port not in {None, 80, 443}):
        raise ValueError("Only public HTTP(S) pages on standard ports are supported")
    addresses = socket.getaddrinfo(parsed.host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    ips = list(dict.fromkeys(address[4][0] for address in addresses))
    if not ips or any(not ip_address(ip).is_global or ip_address(ip).is_multicast for ip in ips):
        raise ValueError("Non-public destination")
    return parsed, ips[0]


def extract_content(html: str) -> tuple[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")
    # Location breadcrumbs often sit beside the h1, outside <main> (e.g. weather sites).
    headings = [heading.parent.get_text(" ", strip=True)[:1200]
                for heading in soup.find_all("h1", limit=2)]
    metadata = [soup.title.get_text(" ", strip=True)] if soup.title else []
    for meta in soup.find_all("meta"):
        key = meta.get("property") or meta.get("name") or ""
        if key in {"description", "og:title", "geo.region", "geo.placename",
                   "place:location:latitude", "place:location:longitude"}:
            metadata.append(f"{key}: {str(meta.get('content', ''))[:500]}")
    published = soup.select_one('meta[property="article:published_time"], meta[name="date"], '
                                'meta[itemprop="datePublished"], time[datetime]')
    published_at = (published.get("content") or published.get("datetime")) if published else None
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg"]):
        tag.decompose()
    for image in soup.select("img[alt]"):
        image.replace_with(str(image.get("alt", "")))
    main = soup.find("main") or soup.find("article") or soup.body or soup
    # Keep line/table order: dates and units must stay associated with forecast values.
    lines = [" ".join(line.split()) for line in main.get_text("\n", strip=True).splitlines()]
    text = "\n".join([*metadata, *headings, *(line for line in lines if line)])
    return text[:MAX_CONTENT_CHARS], str(published_at)[:100] if published_at else None


def fetch_page(result: SearchResult, *, timeout: float) -> SearchResult:
    """Failure on one source preserves its snippet and does not discard other sources."""
    deadline = monotonic() + timeout
    url = str(result.url)
    try:
        # No environment proxies, automatic redirects or cookies shared across sources.
        # A redirect to another host on the same IP must receive its own TLS handshake.
        with httpx.Client(trust_env=False, follow_redirects=False,
                          limits=httpx.Limits(max_keepalive_connections=0)) as client:
            for _ in range(MAX_REDIRECTS + 1):
                parsed, ip = public_target(url)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return result
                target = parsed.copy_with(host=ip)
                headers = {"Host": parsed.netloc.decode("ascii"),
                           "User-Agent": "AIHardcoreEngineer/1.0 (public web retrieval)",
                           "Accept": "text/html,application/xhtml+xml,text/plain",
                           "Accept-Encoding": "identity"}
                # Preserve hostname certificate verification while avoiding DNS rebinding.
                with client.stream("GET", target, headers=headers, timeout=remaining,
                                   extensions={"sni_hostname": parsed.host}) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            return result
                        url = str(parsed.join(location))
                        continue  # Every redirect is separately resolved and validated.
                    response.raise_for_status()
                    media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if media_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                        return result
                    data = bytearray()
                    for block in response.iter_bytes(chunk_size=16_384):
                        if monotonic() > deadline or len(data) + len(block) > MAX_PAGE_BYTES:
                            return result
                        data.extend(block)
                    html = bytes(data).decode(response.encoding or "utf-8", errors="replace")
                    content, published_at = extract_content(html)
                    if len(content.strip()) < 80:
                        return result
                    return SearchResult(**{
                        **result.model_dump(), "url": str(parsed), "content": content, "published_at": published_at,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    })
    except (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError, UnicodeError):
        pass
    return result


def enrich_results(results: list[SearchResult], *, timeout: float) -> list[SearchResult]:
    if not results:
        return results
    selected = results[:MAX_PAGES]
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(selected))) as pool:
        enriched = list(pool.map(lambda result: fetch_page(result, timeout=min(timeout, 6)), selected))
    return [*enriched, *results[MAX_PAGES:]]

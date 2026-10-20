"""Run explicitly: python -m app.rag.ingest [--max-pages 12]."""

import argparse
import base64
import binascii
import json
import re
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
import httpx

from app.config import Settings
from app.schemas import AppError, Chunk

SEED_URLS = [
    "https://site.getnet.com.br/maquininha/get-classica/",
    "https://site.getnet.com.br/maquininha/get-smart/",
    "https://site.getnet.com.br/pix/",
    "https://site.getnet.com.br/crediario/",
    "https://site.getnet.com.br/link-de-pagamento/",
    "https://site.getnet.com.br/duvidas/",
    "https://site.getnet.com.br/get-ajuda-get-conta/o-que-e-a-get-conta/",
    "https://site.getnet.com.br/quando-vale-a-pena-antecipar-as-suas-vendas-no-cartao/",
    "https://site.getnet.com.br/",
    "https://www.getnet.net/",
    "https://www.getnet.net/en",
]
ALLOWED_HOSTS = {"site.getnet.com.br", "www.getnet.net", "getnet.net"}

# Read only literal display data embedded by Getnet's FAQ/technical-sheet widgets.
# This does not execute JavaScript or download scripts.
JS_STRING = r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`'''
JS_TOKENS = re.compile(rf"(?P<string>{JS_STRING})|(?P<comment>//[^\n]*|/\*.*?\*/)", re.DOTALL)
DISPLAY_FIELD = re.compile(rf"\b(?:title|content)\s*:\s*(?P<value>{JS_STRING})(?=\s*[,}}])", re.DOTALL)


def embedded_display_text(soup: BeautifulSoup) -> list[str]:
    texts = []
    for script in soup.find_all("script"):
        code = script.get_text()
        src = script.get("src", "")
        if src.startswith("data:text/javascript;base64,"):
            try:
                code = base64.b64decode(src.split(",", 1)[1], validate=True).decode("utf-8")
            except (ValueError, binascii.Error, UnicodeError):
                continue
        if not any(marker in code for marker in ("faq-content", "FichaTecnicaAtributos(")):
            continue
        section = "Ficha técnica: " if "FichaTecnicaAtributos(" in code else "Perguntas frequentes: "
        code = JS_TOKENS.sub(lambda match: match.group("string") or "", code)
        values = []
        for match in DISPLAY_FIELD.finditer(code):
            value = match.group("value")[1:-1]
            if "${" in value:
                continue  # Computed template values are not static source facts.
            value = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), value)
            value = re.sub(r"\\([\\'\"`/])", r"\1", value)
            value = re.sub(r"\\[nrt]", " ", value)
            text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
            if text:
                values.append(text)
        if values:
            texts.append(section + " ".join(values))
    return texts


def chunk_text(text: str, size: int = 180, overlap: int = 30) -> list[str]:
    """Chunks are measured in words, with a small overlap."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    chunks = []
    # Keep technical sheets and FAQ widgets separate from promotional page copy.
    for section in re.split(r"\n\s*\n", text):
        words = section.split()
        for start in range(0, len(words), size - overlap):
            chunks.append(" ".join(words[start : start + size]))
            if start + size >= len(words):
                break
    return chunks


def is_getnet_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.netloc in ALLOWED_HOSTS


def extract_page(html: str, source: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    embedded = embedded_display_text(soup) if urlparse(source).hostname == "site.getnet.com.br" else []
    links = []
    for anchor in soup.select("a[href]"):
        url = urldefrag(urljoin(source, anchor["href"]))[0]
        if (is_getnet_url(url) and not urlparse(url).query
                and not urlparse(url).path.startswith(("/wp-", "/api/"))):
            links.append(url)
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "button", "input", "select", "svg"]):
        tag.decompose()
    content = soup.find("main") or soup.find("article") or soup.body or soup
    text = " ".join(content.stripped_strings)
    # Append the widget's literal question/answer text, including HTML-hidden FAQs.
    return "\n\n".join(part for part in [text, *embedded] if part), list(dict.fromkeys(links))


def ingest(output: Path, *, max_pages: int = 12, timeout: float = 15) -> dict:
    if not 2 <= max_pages <= 20:
        raise ValueError("max_pages deve estar entre 2 e 20.")
    pending = list(SEED_URLS)
    visited = set()
    chunks = []
    failures = []
    loaded_sources = set()
    pages_loaded = 0
    with httpx.Client(
        timeout=timeout, follow_redirects=True,
        headers={"User-Agent": "AIHardcoreEngineer/1.0 (educational RAG)"},
    ) as client:
        while pending and len(visited) < max_pages:
            url = pending.pop(0)
            if url in visited:
                continue
            visited.add(url)
            try:
                response = client.get(url)
                response.raise_for_status()
                source = str(response.url)
                if not is_getnet_url(source):
                    raise ValueError("Redirecionamento para fora do domínio Getnet.")
                if source in loaded_sources:
                    continue
                if "text/html" not in response.headers.get("content-type", ""):
                    raise ValueError("A página não retornou HTML.")
                text, links = extract_page(response.text, source)
                if len(text.split()) < 20:
                    raise ValueError("Página sem texto suficiente para ingestão.")
                chunks.extend(Chunk(text=part, source=source) for part in chunk_text(text))
                loaded_sources.add(source)
                pages_loaded += 1
                pending.extend(link for link in links if link not in visited and link not in pending)
            except (httpx.HTTPError, ValueError) as exc:
                failures.append({"url": url, "error": str(exc)})
    if not chunks:
        raise AppError(503, f"Nenhuma página Getnet foi carregada; base anterior preservada. {failures}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    return {"pages_loaded": pages_loaded, "chunks": len(chunks), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Carrega até 20 páginas públicas Getnet para RAG local.")
    parser.add_argument("--max-pages", type=int, default=12, choices=range(2, 21))
    args = parser.parse_args()
    settings = Settings.from_env()
    try:
        report = ingest(settings.data_dir / "rag" / "chunks.json", max_pages=args.max_pages,
                        timeout=settings.http_timeout_seconds)
    except (AppError, OSError) as exc:
        parser.exit(1, f"Falha na ingestão: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

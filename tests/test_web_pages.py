import importlib
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import Mock

import httpx
import pytest

from app.schemas import SearchResult
from app.agents.knowledge import KnowledgeAgent
from app.tools import web_pages

# Preserve the real resolver before the autouse no-network fixture replaces it.
public_target = web_pages.public_target


def result(url="https://example.com/forecast"):
    return SearchResult(title="Forecast", snippet="Forecast page", url=url)


def transport(monkeypatch, handler):
    real_client = httpx.Client
    monkeypatch.setattr(web_pages.httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    resolver = Mock(side_effect=lambda url: (httpx.URL(url), "93.184.216.34"))
    monkeypatch.setattr(web_pages, "public_target", resolver)
    return resolver


def test_page_content_keeps_forecast_values_dates_and_source(monkeypatch):
    def respond(request):
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text="""
            <title>Porto Alegre weather</title>
            <meta property="place:location:latitude" content="-30.03">
            <meta property="article:published_time" content="2026-10-29T07:00:00-03:00">
            <header><h1>Porto Alegre</h1><span>Rio Grande do Sul, Brazil</span></header>
            <nav>Navigation noise</nav><main><h2>Porto Alegre forecast for 2026-10-30</h2>
            <table><tr><td>Minimum</td><td>18 °C</td></tr>
            <tr><td>Maximum</td><td>26 °C</td></tr></table>
            <p>Probability of precipitation: 60 percent.</p><img alt="Light rain">
            <script>execute_me()</script></main>""")

    transport(monkeypatch, respond)
    enriched = web_pages.fetch_page(result(), timeout=2)
    assert "2026-10-30" in enriched.content
    assert "18 °C" in enriched.content
    assert "60 percent" in enriched.content
    assert "Rio Grande do Sul, Brazil" in enriched.content
    assert "place:location:latitude: -30.03" in enriched.content
    assert "Light rain" in enriched.content
    assert "Navigation noise" not in enriched.content
    assert "execute_me" not in enriched.content
    assert enriched.published_at == "2026-10-29T07:00:00-03:00"
    assert enriched.retrieved_at
    assert str(enriched.url) == "https://example.com/forecast"
    assert enriched.snippet == "Forecast page"


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1",
                                     "0.0.0.0", "::ffff:127.0.0.1", "100.64.0.1", "224.0.0.1", "ff02::1"])
def test_private_dns_addresses_are_rejected(monkeypatch, address):
    monkeypatch.setattr(web_pages.socket, "getaddrinfo", lambda *args, **kwargs: [
        (2, 1, 6, "", (address, 443))])
    with pytest.raises(ValueError):
        public_target("https://example.com")


def test_mixed_public_private_dns_is_rejected(monkeypatch):
    monkeypatch.setattr(web_pages.socket, "getaddrinfo", lambda *args, **kwargs: [
        (2, 1, 6, "", ("93.184.216.34", 443)), (2, 1, 6, "", ("10.0.0.1", 443))])
    with pytest.raises(ValueError):
        public_target("https://example.com")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/file", "https://a:b@example.com/",
                                 "http://example.com:8080/", "http://example.com:22/"])
def test_invalid_page_targets_rejected_without_dns(monkeypatch, url):
    resolve = Mock(side_effect=AssertionError("DNS must not be called"))
    monkeypatch.setattr(web_pages.socket, "getaddrinfo", resolve)
    with pytest.raises(ValueError):
        public_target(url)
    resolve.assert_not_called()


def test_redirect_to_private_network_preserves_snippet(monkeypatch):
    resolver = transport(monkeypatch, lambda request: httpx.Response(
        302, headers={"location": "http://169.254.169.254/latest/meta-data/"}))
    resolver.side_effect = [(httpx.URL("https://example.com/forecast"), "93.184.216.34"),
                            ValueError("Non-public destination")]
    original = result()
    assert web_pages.fetch_page(original, timeout=2) is original
    assert resolver.call_count == 2


def test_public_redirect_preserves_final_source_url(monkeypatch):
    def respond(request):
        if request.url.path == "/forecast":
            return httpx.Response(302, headers={"location": "/updated"})
        return httpx.Response(200, headers={"content-type": "text/html"},
                              text="<main>Updated forecast: " + "weather data " * 20 + "</main>")

    resolver = transport(monkeypatch, respond)
    enriched = web_pages.fetch_page(result(), timeout=2)
    assert str(enriched.url) == "https://example.com/updated"
    assert enriched.content.startswith("Updated forecast")
    assert resolver.call_count == 2


@pytest.mark.parametrize("kind", ["timeout", "http_error", "pdf", "oversized", "redirect_loop", "invalid_redirect"])
def test_unavailable_or_unsupported_pages_preserve_search_snippets(monkeypatch, kind):
    def respond(request):
        if kind == "timeout":
            raise httpx.ReadTimeout("unavailable")
        if kind == "http_error":
            return httpx.Response(403)
        if kind == "redirect_loop":
            return httpx.Response(302, headers={"location": "/forecast"})
        if kind == "invalid_redirect":
            return httpx.Response(302, headers={"location": "https://[invalid"})
        return httpx.Response(200, headers={"content-type": "application/pdf" if kind == "pdf" else "text/html"},
                              content=b"x" * (web_pages.MAX_PAGE_BYTES + 1 if kind == "oversized" else 100))

    resolver = transport(monkeypatch, respond)
    original = result()
    assert web_pages.fetch_page(original, timeout=2) is original
    if kind == "redirect_loop":
        assert resolver.call_count == web_pages.MAX_REDIRECTS + 1


def test_extracted_context_is_bounded_and_does_not_fabricate_publication_date():
    content, published_at = web_pages.extract_content("<main>" + "weather " * 5000 + "</main>")
    assert len(content) == web_pages.MAX_CONTENT_CHARS
    assert published_at is None


def test_page_downloads_are_bounded_and_all_search_results_are_covered(monkeypatch):
    fetched = Mock(side_effect=lambda item, **kwargs: item.model_copy(update={"content": "Downloaded"}))
    monkeypatch.setattr(web_pages, "fetch_page", fetched)
    originals = [result(f"https://example.com/{n}") for n in range(7)]
    enriched = web_pages.enrich_results(originals, timeout=15)
    assert fetched.call_count == 5
    assert all(call.kwargs["timeout"] == 6 for call in fetched.call_args_list)
    assert [item.content for item in enriched] == ["Downloaded"] * 5 + [""] * 2
    assert [item.url for item in enriched] == [item.url for item in originals]


def test_search_passes_normalized_results_to_page_enrichment(monkeypatch):
    search = importlib.import_module("app.tools.web_search")
    monkeypatch.setattr(search, "DDGS", Mock(return_value=Mock(text=Mock(return_value=[
        {"title": "Forecast", "body": "Forecast page", "href": "https://example.com/forecast"}]))))
    enriched = result().model_copy(update={"content": "Actual forecast values"})
    enrich = Mock(return_value=[enriched])
    monkeypatch.setattr(search, "enrich_results", enrich)
    assert search.web_search("Porto Alegre tomorrow", timeout=9) == [enriched]
    enrich.assert_called_once_with([result()], timeout=9)


def test_weather_agent_receives_page_values_and_date_checks_without_extra_llm_calls(monkeypatch, gemini, settings):
    search = importlib.import_module("app.tools.web_search")
    monkeypatch.setattr(search, "DDGS", Mock(return_value=Mock(text=Mock(return_value=[
        {"title": "Forecast", "body": "Tomorrow forecast page", "href": "https://example.com/forecast"}]))))
    transport(monkeypatch, lambda request: httpx.Response(
        200, headers={"content-type": "text/html"}, text="""
        <header><h1>Porto Alegre</h1><p>Rio Grande do Sul, Brazil</p></header>
        <main>Forecast for 2026-10-30. Minimum 18 °C, maximum 26 °C.
        Probability of rain 60 percent. Published 2026-10-29.</main>"""))
    gemini.generate.return_value = json.dumps({"sufficient_context": True,
        "answer": "For Porto Alegre, RS, on 2026-10-30: 18–26 °C and 60% chance of rain. https://example.com/forecast"})
    answer = KnowledgeAgent(gemini, settings).answer(
        "What's the weather forecast in Porto Alegre tomorrow?", "knowledge_web")
    assert answer.status == "answered"
    assert not answer.requires_human
    assert answer.tools_used == ["web_search"]
    assert answer.sources == ["https://example.com/forecast"]
    assert answer.evidence[0]["content"]
    prompt = gemini.generate.call_args.args[0]
    assert "18 °C" in prompt and "60 percent" in prompt
    assert "Rio Grande do Sul" in prompt
    assert "never proof of the date of the facts" in prompt
    assert "never combine forecasts for different locations" in prompt
    tomorrow = datetime.now(ZoneInfo("America/Sao_Paulo")).date() + timedelta(days=1)
    assert f"Tomorrow: {tomorrow} ({tomorrow.strftime('%A')})" in prompt
    gemini.generate.assert_called_once()

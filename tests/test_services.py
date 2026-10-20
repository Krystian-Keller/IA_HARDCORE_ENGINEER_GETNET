import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import Client as GenaiClient, errors

from app.config import Settings
from app.schemas import AppError, GuardrailResult, KnowledgeAnswer, RouteDecision, ToolDecision
from app.services.gemini import Gemini
from app.tools.web_search import web_search


def sdk_mock(monkeypatch):
    factory = MagicMock()
    client = factory.return_value.__enter__.return_value
    monkeypatch.setattr("app.services.gemini.genai.Client", factory)
    return factory, client


def test_gemini_uses_env_configuration_and_structured_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "7")
    factory, client = sdk_mock(monkeypatch)
    client.models.generate_content.return_value = SimpleNamespace(text='{"route":"knowledge_web"}')
    result = Gemini(Settings.from_env()).generate("classify", schema=RouteDecision)
    assert result == '{"route":"knowledge_web"}'
    assert factory.call_args.kwargs["api_key"] == "fake-test-key"
    assert factory.call_args.kwargs["http_options"].timeout == 7000
    assert factory.call_args.kwargs["http_options"].retry_options.attempts == 1
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_json_schema == RouteDecision.model_json_schema()
    assert kwargs["config"].response_schema is None


@pytest.mark.parametrize("schema,answer", [
    (GuardrailResult, '{"category":"supported","reason":"normal request"}'),
    (RouteDecision, '{"route":"knowledge_web"}'),
    (ToolDecision, '{"tools":["get_customer_activity"]}'),
    (KnowledgeAnswer, '{"sufficient_context":true,"answer":"Grounded answer."}'),
    (None, "A grounded answer."),
])
def test_real_sdk_serializes_supported_json_schema_without_network(monkeypatch, schema, answer):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        config = payload["generationConfig"]
        assert "responseSchema" not in config
        if schema is not None:
            assert config["responseMimeType"] == "application/json"
            assert config["responseJsonSchema"] == schema.model_json_schema()
            assert config["responseJsonSchema"]["additionalProperties"] is False
            assert "additional_properties" not in json.dumps(payload)
        else:
            assert "responseJsonSchema" not in config
        return httpx.Response(200, json={"candidates": [{
            "content": {"role": "model", "parts": [{"text": answer}]}, "finishReason": "STOP"
        }]})

    def make_client(**kwargs):
        options = kwargs["http_options"].model_copy(update={
            "httpx_client": httpx.Client(transport=httpx.MockTransport(respond))
        })
        return GenaiClient(api_key=kwargs["api_key"], http_options=options)

    monkeypatch.setattr("app.services.gemini.genai.Client", make_client)
    assert Gemini(Settings(gemini_api_key="fake")).generate("Test", schema=schema) == answer
    assert len(requests) == 1


def test_missing_key_is_controlled():
    with pytest.raises(AppError) as error:
        Gemini(Settings()).generate("hello")
    assert error.value.status_code == 503


@pytest.mark.parametrize("failure,status,detail", [
    (httpx.ReadTimeout("timeout"), 504, "Tempo limite"),
    (httpx.ConnectError("offline"), 502, "Falha de conexão"),
    (errors.ClientError(400, {"error": {"message": "bad request"}}), 502, "parâmetros enviados"),
    (errors.ClientError(401, {"error": {"message": "invalid key"}}), 502, "GEMINI_API_KEY"),
    (errors.ClientError(403, {"error": {"message": "denied"}}), 502, "permissões"),
    (errors.ClientError(404, {"error": {"message": "model missing"}}), 502, "GEMINI_MODEL"),
    (errors.ClientError(429, {"error": {"message": "quota"}}), 429, "cota"),
    (errors.ServerError(500, {"error": {"message": "internal"}}), 502, "Não foi possível"),
    (errors.ServerError(503, {"error": {"message": "high demand"}}), 503, "alta demanda"),
    (errors.ServerError(504, {"error": {"message": "timeout"}}), 504, "Tempo limite"),
])
def test_gemini_failures(monkeypatch, failure, status, detail):
    _, client = sdk_mock(monkeypatch)
    client.models.generate_content.side_effect = failure
    with pytest.raises(AppError) as error:
        Gemini(Settings(gemini_api_key="fake")).generate("hello")
    assert error.value.status_code == status
    assert detail in error.value.detail


def test_provider_error_does_not_expose_credentials_or_raw_payload(monkeypatch):
    _, client = sdk_mock(monkeypatch)
    key = "fake-private-credential"
    client.models.generate_content.side_effect = errors.ClientError(
        400, {"error": {"message": f"Rejected request containing {key} and private prompt data"}}
    )
    with pytest.raises(AppError) as error:
        Gemini(Settings(gemini_api_key=key)).generate("private prompt data")
    assert key not in error.value.detail
    assert "private prompt data" not in error.value.detail


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_or_blocked_gemini_response(monkeypatch, text):
    _, client = sdk_mock(monkeypatch)
    client.models.generate_content.return_value = SimpleNamespace(text=text)
    with pytest.raises(AppError) as error:
        Gemini(Settings(gemini_api_key="fake")).generate("hello")
    assert error.value.status_code == 502


def test_search_normalizes_results_and_uses_duckduckgo(monkeypatch):
    factory = MagicMock()
    factory.return_value.text.return_value = [
        {"title": "Result", "body": "Context", "href": "https://example.com/news"},
        {"title": "Invalid", "body": "No URL", "href": "javascript:alert(1)"},
        {"title": "", "body": "Empty title", "href": "https://example.com"},
    ]
    monkeypatch.setattr("app.tools.web_search.DDGS", factory)
    results = web_search("query", max_results=3, timeout=8)
    assert len(results) == 1
    assert results[0].snippet == "Context"
    assert str(results[0].url) == "https://example.com/news"
    factory.assert_called_once_with(timeout=8)
    factory.return_value.text.assert_called_once_with("query", max_results=3, backend="duckduckgo")


def test_search_failure_is_controlled(monkeypatch):
    monkeypatch.setattr("app.tools.web_search.DDGS", MagicMock(side_effect=RuntimeError("blocked")))
    with pytest.raises(AppError) as error:
        web_search("query")
    assert error.value.status_code == 502

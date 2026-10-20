import json

import pytest

from app.schemas import AppError, RouteDecision
from tests.conftest import ACCEPTANCE_CASES


@pytest.mark.parametrize("message,route", ACCEPTANCE_CASES)
def test_classifies_with_validated_gemini_json(router, gemini, message, route):
    gemini.generate.return_value = json.dumps({"route": route})
    assert router.classify(message) == route
    prompt = gemini.generate.call_args.args[0]
    assert json.dumps(message, ensure_ascii=False) in prompt
    assert gemini.generate.call_args.kwargs["schema"] is RouteDecision


@pytest.mark.parametrize("raw", [
    "not json", '{"route":"unknown"}', '{}', '{"route":"knowledge_web","extra":true}',
    '```json\n{"route":"knowledge_web"}\n```',
])
def test_rejects_invalid_routes_before_dispatch(router, gemini, raw):
    gemini.generate.return_value = raw
    with pytest.raises(AppError) as error:
        router.classify("hello")
    assert error.value.status_code == 502

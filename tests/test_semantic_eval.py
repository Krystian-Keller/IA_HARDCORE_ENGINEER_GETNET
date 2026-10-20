import pytest
from pydantic import ValidationError

from evals.quality import JudgeResult, SemanticConstraints, check_constraints, judge_answer


def test_semantic_constraints_check_concepts_forbidden_claims_language_and_evidence():
    source = "https://site.getnet.com.br/pix/"
    constraints = SemanticConstraints(required_concepts=[["Pix"],["bank", "banco"]],
        forbidden_claims=["guaranteed transfer"], require_citations=True, expected_language="en")
    answer = f"You can use Pix with your bank. {source}"
    assert all(check_constraints(answer, [source], constraints, [{"source":source}]).values())
    assert not all(check_constraints("Guaranteed transfer", [source], constraints).values())
    assert not check_constraints(answer, [source], constraints, [{"source":"https://evil.example/"}])["citations"]
    assert not check_constraints(answer + " https://evil.example/", [source], constraints)["citations"]
    assert not check_constraints("Você pode usar o Pix no seu banco.", [source], constraints)["language_heuristic"]


def test_judge_receives_question_answer_evidence_constraints(gemini):
    gemini.generate.return_value = '{"groundedness":5,"relevance":4,"correctness":5,"unsupported_claims":[],"pass":true}'
    result = judge_answer(gemini, "Question", "Answer", [{"text":"Relevant evidence"}],
                          SemanticConstraints(required_concepts=[["Pix"]]))
    assert result.passed
    assert gemini.generate.call_args.kwargs["schema"] is JudgeResult
    for text in ["Question", "Answer", "Relevant evidence", "Pix"]:
        assert text in gemini.generate.call_args.args[0]


@pytest.mark.parametrize("raw", ["not json", "{}",
    '{"groundedness":6,"relevance":4,"correctness":5,"unsupported_claims":[],"pass":true}',
    '{"groundedness":true,"relevance":4,"correctness":5,"unsupported_claims":[],"pass":true}',
    '{"groundedness":1,"relevance":4,"correctness":5,"unsupported_claims":[],"pass":true}',
    '{"groundedness":5,"relevance":4,"correctness":5,"unsupported_claims":["invention"],"pass":true}',
])
def test_malformed_or_inconsistent_judge_is_rejected(gemini, raw):
    gemini.generate.return_value = raw
    with pytest.raises(ValidationError):
        judge_answer(gemini, "Question", "Answer", [], SemanticConstraints())


@pytest.mark.parametrize("value", [{"required_concepts":[[]]}, {"required_concepts":[[""]]},
                                    {"expected_language":"xx"}, {"forbidden_claims":[""]}])
def test_semantic_dataset_constraints_are_validated(value):
    with pytest.raises(ValidationError):
        SemanticConstraints.model_validate(value)


def test_invalid_live_judge_output_fails_case_without_leaking_provider_error(gemini):
    import httpx
    from unittest.mock import Mock
    from uuid import uuid4
    from evals.run import EvaluationCase, evaluate_case
    case = EvaluationCase(name="judge", message="Pix?", user_id="a", expected_route="knowledge_getnet",
                          expected_tools=["getnet_rag"])
    body = {"request_id":str(uuid4()), "conversation_id":str(uuid4()), "answer":"Pix is available.",
            "route":"knowledge_getnet", "guardrail_category":"supported", "status":"answered",
            "agents_used":["router","knowledge"], "tools_used":["getnet_rag"], "sources":[]}
    store = Mock()
    store.latest_evidence.return_value = [{"text":"Pix is available."}]
    gemini.generate.return_value = "invalid PRIVATE provider payload"
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))) as client:
        result = evaluate_case(case, client, "http://test", judge=gemini, store=store)
    assert not result["passed"] and result["actual"]["judge_error"] == "ValidationError"
    assert "PRIVATE" not in str(result)


@pytest.mark.parametrize("message,language", [
    ("What's the difference between Get Clássica and Get Smart?", "English"),
    ("Do I need a bank account to receive sales via Pix?", "English"),
    ("How many installments with crediário?", "English"),
    ("É a Smart.", "Portuguese"), ("Minha maquininha não conecta.", "Portuguese"),
])
def test_explicit_language_hint_uses_question_instead_of_source_terms(message, language):
    from app.language import answer_language
    assert answer_language(message) == language

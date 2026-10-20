import json

import pytest

from tests.conftest import SUPPORTED


@pytest.mark.parametrize("key,user,status", [(None, "cliente1988",401), ("bad", "cliente1988",401),
                                            ("demo-a", "other",403)])
def test_demo_identity_rejects_unauthorized_callers(client, router, gemini, key, user, status):
    from pydantic import SecretStr
    router.settings.demo_api_keys = {"demo-a":"cliente1988"}
    router.settings.reviewer_api_key = SecretStr("reviewer")
    response = client.post("/chat", json={"message":"My records", "user_id":user},
                           headers={"X-API-Key":key} if key else {})
    assert response.status_code == status
    gemini.generate.assert_not_called()


def test_demo_identity_and_separate_reviewer_key(client, router, gemini):
    from pydantic import SecretStr
    router.settings.demo_api_keys = {"demo-a":"cliente1988"}
    router.settings.reviewer_api_key = SecretStr("reviewer")
    gemini.generate.return_value = SUPPORTED
    body = client.post("/chat", headers={"X-API-Key":"demo-a"},
        json={"message":"human support", "user_id":"cliente1988"}).json()
    url = f"/handoffs/{body['handoff_id']}/reply"
    gemini.generate.reset_mock()
    assert client.post(url, headers={"X-API-Key":"demo-a"}, json={"response":"Note"}).status_code == 401
    gemini.generate.assert_not_called()
    gemini.generate.return_value = "Reviewed answer"
    assert client.post(url, headers={"X-Reviewer-Key":"reviewer"}, json={"response":"Note"}).status_code == 200


def test_readiness_reports_local_resources_without_network(client, router, gemini, settings):
    from pydantic import SecretStr
    assert client.get("/health").json() == {"status":"ok"}
    response = client.get("/ready")
    assert response.status_code == 503
    resources = response.json()["resources"]
    assert resources["conversation_database"] == "available"
    assert resources["rag_index"] == "missing" and resources["gemini_configuration"] == "missing"
    assert resources["gemini_connectivity"] == "not_checked"
    settings.gemini_api_key = SecretStr("configured-demo")
    corpus = settings.data_dir / "rag/chunks.json"
    corpus.parent.mkdir()
    corpus.write_text(json.dumps([{"text":"Getnet Pix payments", "source":"https://site.getnet.com.br/pix/"}]))
    assert client.get("/ready").status_code == 200
    router.store.path = settings.data_dir / "missing/invalid.db"
    assert client.get("/ready").json()["resources"]["conversation_database"] == "missing"
    gemini.generate.assert_not_called()


@pytest.mark.parametrize("reviewer", ["", "PRIVATE-TEST-KEY"])
def test_identity_configuration_requires_separate_reviewer_credential(reviewer):
    from pydantic import ValidationError
    from app.config import Settings
    with pytest.raises(ValidationError) as error:
        Settings(demo_api_keys={"PRIVATE-TEST-KEY":"synthetic-user"}, reviewer_api_key=reviewer)
    assert "PRIVATE-TEST-KEY" not in str(error.value)


def test_unusable_database_is_reported_before_first_chat(client, router, settings, gemini):
    # Lazy store initialization keeps liveness and readiness usable during local storage failure.
    settings.data_dir = settings.data_dir / "not-a-directory"
    settings.data_dir.write_text("file in place of a data directory")
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["resources"]["conversation_database"] == "missing"
    gemini.generate.assert_not_called()

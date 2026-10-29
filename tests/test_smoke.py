import json
from uuid import uuid4

import httpx
import pytest

from evals.smoke import run_smoke


@pytest.mark.parametrize("resume_matches", [True, False])
def test_smoke_validates_continuity_and_local_resume_without_printing_private_data(capsys, resume_matches):
    cid, handoff_cid, hid = (str(uuid4()) for _ in range(3))
    messages = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status":"ok"})
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status":"ready"})
        body = json.loads(request.content)
        if request.url.path == "/chat":
            messages.append(body)
            index = len(messages)
            if index == 4:
                return httpx.Response(200, json={"status":"awaiting_human", "requires_human":True,
                    "conversation_id":handoff_cid, "handoff_id":hid})
            return httpx.Response(200, json={"status":"answered", "conversation_id":cid,
                "answer":"PRIVATE-CUSTOMER-ANSWER", "agents_used":["customer_support","knowledge","synthesis"]})
        assert request.headers["X-Reviewer-Key"] == "PRIVATE-REVIEWER-KEY"
        return httpx.Response(200, json={"status":"answered", "requires_human":False,
            "conversation_id":handoff_cid if resume_matches else str(uuid4())})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        result = run_smoke(client, reviewer_key="PRIVATE-REVIEWER-KEY")
    assert result == {"passed":7 if resume_matches else 6, "failed":0 if resume_matches else 1, "total":7}
    assert messages[1]["conversation_id"] == cid
    assert "PRIVATE" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["network", "json", "object", "http"])
def test_smoke_reports_all_seven_stages_despite_network_or_invalid_api_output(failure, capsys):
    def handler(request):
        if failure == "network":
            raise httpx.ConnectError("PRIVATE network detail", request=request)
        if failure == "json":
            return httpx.Response(502, text="PRIVATE proxy response")
        if failure == "object":
            return httpx.Response(200, json=["PRIVATE invalid output"])
        return httpx.Response(503, json={"detail":"PRIVATE provider detail"})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as client:
        assert run_smoke(client) == {"passed":0, "failed":7, "total":7}
    output = capsys.readouterr().out
    assert "PRIVATE" not in output
    assert '"check": "resume"' in output

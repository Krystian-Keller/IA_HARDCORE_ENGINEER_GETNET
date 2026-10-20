from pathlib import Path
from shutil import copyfile
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.agents.customer_support import CustomerSupportAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.router import RouterAgent
from app.config import Settings
from app.main import app, get_router
from app.services.gemini import Gemini

SUPPORTED = '{"category":"supported","reason":"normal supported request"}'


@pytest.fixture(autouse=True)
def block_live_services(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must mock external services")

    monkeypatch.setattr("app.services.gemini.genai.Client", forbidden)
    monkeypatch.setattr("app.tools.web_search.DDGS", forbidden)


@pytest.fixture
def settings(tmp_path):
    copyfile(Path(__file__).resolve().parents[1] / "data/customers.json", tmp_path / "customers.json")
    return Settings(data_dir=tmp_path)


@pytest.fixture
def gemini():
    return Mock(spec=Gemini)


@pytest.fixture
def router(gemini, settings):
    return RouterAgent(gemini, KnowledgeAgent(gemini, settings), CustomerSupportAgent(gemini, settings))


@pytest.fixture
def client(router):
    app.dependency_overrides[get_router] = lambda: router
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        get_router.cache_clear()


ACCEPTANCE_CASES = [
    ("What's the difference between the Get Clássica and the Get Smart?", "knowledge_getnet"),
    ("What's the weather forecast in Porto Alegre tomorrow?", "knowledge_web"),
    ("When will the money from yesterday's sales be deposited?", "customer_support"),
    ("Do I need a bank account to receive my sales via Pix?", "knowledge_getnet"),
    ("My card machine won't connect to the internet, what should I do?", "customer_support"),
    ("How does receivables advance (antecipação) work with Getnet?", "knowledge_getnet"),
    ("What's the euro exchange rate today?", "knowledge_web"),
    ("My card machine is showing a transaction decline error.", "customer_support"),
    ("How many installments can I split a sale into with the crediário?", "knowledge_getnet"),
    ("Can I sell through WhatsApp using the Payment Link?", "knowledge_getnet"),
]

import base64
import json

import httpx
import pytest

from app.rag import ingest as ingestion
from app.rag.retriever import Retriever
from app.schemas import AppError, Chunk


def test_retrieves_relevant_local_chunks_without_network(tmp_path):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps([
        {"text": "Pix is an instant payment service with QR codes.", "source": "https://www.getnet.net/pix"},
        {"text": "Smart card machines print receipts and connect via Wi-Fi.", "source": "https://www.getnet.net/machines"},
        {"text": "Antecipação permite antecipar recebíveis de vendas.", "source": "https://www.getnet.net/advance"},
    ]), encoding="utf-8")
    retriever = Retriever.from_file(path)
    assert str(retriever.retrieve("instant Pix QR", top_k=1)[0].source).endswith("/pix")
    assert str(retriever.retrieve("card machines receipts", top_k=1)[0].source).endswith("/machines")
    assert str(retriever.retrieve("antecipacao recebiveis", top_k=1)[0].source).endswith("/advance")
    assert retriever.retrieve("astronomy") == []
    assert retriever.retrieve("Pix", top_k=0) == []


@pytest.mark.parametrize("content", [None, "not json", "[]", '[{"text":"missing source"}]'])
def test_missing_or_invalid_corpus_is_explicit(tmp_path, content):
    path = tmp_path / "chunks.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(AppError) as error:
        Retriever.from_file(path)
    assert error.value.status_code == 503


def test_unusable_vocabulary():
    with pytest.raises(AppError):
        Retriever([Chunk(text="a !", source="https://www.getnet.net/")])


def test_chunking_preserves_words_and_overlap():
    assert ingestion.chunk_text("one two three four five six", size=4, overlap=2) == [
        "one two three four", "three four five six"
    ]
    assert ingestion.chunk_text("") == []
    assert ingestion.chunk_text("Promotion terms\n\nTechnical sheet", size=4, overlap=2) == [
        "Promotion terms", "Technical sheet"
    ]
    with pytest.raises(ValueError):
        ingestion.chunk_text("hello", size=2, overlap=2)


def test_html_extraction_and_domain_limits():
    html = '<nav>navigation</nav><script>bad()</script><main>Pix payment information</main>' \
           '<a href="/en">English</a><a href="https://evil.example/">external</a>' \
           '<a href="https://www.getnet.net.evil.example/">lookalike</a>' \
           '<a href="/en?search=foo">query</a>'
    text, links = ingestion.extract_page(html, "https://www.getnet.net/")
    assert text == "Pix payment information"
    assert links == ["https://www.getnet.net/en"]


def mock_downloads(monkeypatch, handler):
    original_client = httpx.Client
    monkeypatch.setattr(ingestion.httpx, "Client", lambda **kwargs: original_client(
        **kwargs, transport=httpx.MockTransport(handler)
    ))


def test_ingest_limits_crawl_and_keeps_sources(tmp_path, monkeypatch):
    fetched = []

    def handler(request):
        fetched.append(str(request.url))
        html = '<main>' + 'Getnet Pix payments and card machine services. ' * 30 + '</main>'
        html += ''.join(f'<a href="/page-{n}">page</a>' for n in range(10))
        html += '<a href="https://other.example">external</a>'
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    mock_downloads(monkeypatch, handler)
    output = tmp_path / "rag/chunks.json"
    report = ingestion.ingest(output, max_pages=3)
    assert len(fetched) == 3
    assert fetched == ingestion.SEED_URLS[:3]
    assert report["pages_loaded"] == 3
    assert report["failures"] == []
    assert Retriever.from_file(output).retrieve("Pix")
    assert {chunk["source"] for chunk in json.loads(output.read_text())} == set(fetched)


def test_failed_ingestion_preserves_existing_corpus(tmp_path, monkeypatch):
    output = tmp_path / "chunks.json"
    output.write_text("previous corpus")
    mock_downloads(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(AppError, match="preservada"):
        ingestion.ingest(output, max_pages=2)
    assert output.read_text() == "previous corpus"


def test_redirected_seed_pages_do_not_duplicate_chunks(tmp_path, monkeypatch):
    def handler(request):
        if request.url.path != "/en/":
            return httpx.Response(302, headers={"location": "https://www.getnet.net/en/"})
        return httpx.Response(200, text='<main>' + 'Getnet payment products ' * 30 + '</main>',
                              headers={"content-type": "text/html"})

    mock_downloads(monkeypatch, handler)
    output = tmp_path / "chunks.json"
    report = ingestion.ingest(output, max_pages=2)
    assert report["pages_loaded"] == 1
    assert report["chunks"] == 1
    assert json.loads(output.read_text())[0]["source"] == "https://www.getnet.net/en/"


def test_partial_ingestion_reports_failure(tmp_path, monkeypatch):
    def handler(request):
        if str(request.url) == ingestion.SEED_URLS[1]:
            return httpx.Response(403)
        return httpx.Response(200, text='<main>' + 'Getnet products ' * 30 + '</main>',
                              headers={"content-type": "text/html"})

    mock_downloads(monkeypatch, handler)
    report = ingestion.ingest(tmp_path / "chunks.json", max_pages=2)
    assert report["pages_loaded"] == 1
    assert report["failures"][0]["url"] == ingestion.SEED_URLS[1]


@pytest.mark.parametrize("url,allowed", [
    ("https://site.getnet.com.br/pix/", True),
    ("https://site.getnet.com.br/maquininha/get-smart/", True),
    ("https://www.getnet.net/en/", True),
    ("https://site.getnet.com.br.evil.example/pix/", False),
    ("https://evil.example/site.getnet.com.br", False),
    ("https://user@site.getnet.com.br/pix/", False),
    ("https://site.getnet.com.br:8443/pix/", False),
    ("http://site.getnet.com.br/pix/", False),
])
def test_brazilian_domain_allowlist(url, allowed):
    assert ingestion.is_getnet_url(url) is allowed


@pytest.mark.parametrize("encoded", [False, True])
def test_static_widget_extraction_keeps_facts_and_discards_commented_or_computed_data(encoded):
    script = r'''
    // faq-content; FichaTecnicaAtributos(
    const fields = [
      // {title: "Outdated offer", content: "Old tariff"},
      /* {title: "Obsolete rule", content: "Wrong feature"}, */
      {title: "Teclado f\u00edsico", content: '<p>Visor touchscreen e colorido</p>'},
      {title: 'Como habilitar?', content: `Use https://site.getnet.com.br/pix/`},
      {title: 'Dynamic template', content: `${executeSomething()}`},
      {title: computeTitle(), content: 'literal value'}
    ];
    '''
    if encoded:
        data = base64.b64encode(script.encode()).decode()
        html = f'<main>Product details</main><script src="data:text/javascript;base64,{data}"></script>'
    else:
        html = f'<main>Product details</main><script>{script}</script>'
    text, _ = ingestion.extract_page(html, ingestion.SEED_URLS[0])
    assert "Teclado físico" in text
    assert "Visor touchscreen e colorido" in text
    assert "https://site.getnet.com.br/pix/" in text
    assert all(value not in text for value in ("Outdated", "Old tariff", "Obsolete", "Wrong feature",
                                               "executeSomething", "computeTitle"))


def test_widget_extraction_ignores_other_scripts_and_invalid_base64():
    html = '''<main>Pix</main>
      <script>const data = {title: "Unrelated internal setting"};</script>
      <script src="data:text/javascript;base64,!!!invalid!!!"></script>
      <a href="/pix/#faq">Pix</a><a href="/pix/?q=test">Query</a>
      <a href="/wp-content/widget.js">Asset</a><a href="/api/data">API</a>'''
    text, links = ingestion.extract_page(html, "https://site.getnet.com.br/")
    assert text == "Pix"
    assert links == ["https://site.getnet.com.br/pix/"]


@pytest.mark.parametrize("question", [
    "What's the difference between the Get Clássica and the Get Smart?",
    "Qual a diferença entre a Get Clássica e a Get Smart?",
])
def test_comparison_recovers_features_of_both_products_after_ingestion(tmp_path, monkeypatch, question):
    features = {
        "/maquininha/get-classica/": "Teclado físico. Visor touchscreen e colorido. Bateria de até 12 horas.",
        "/maquininha/get-smart/": "Teclado touch. Visor touchscreen e colorido. Aplicativos e controle de estoque.",
    }

    def handler(request):
        fact = features.get(request.url.path, "Serviços globais para empresas e restaurantes.")
        # The features live in inline widget data, as on the public Brazilian pages.
        script = "/* FichaTecnicaAtributos( */ const attributes = [{title: 'Ficha técnica', content: '" + fact + "'}];"
        html = '<main>' + 'Oferta Get Clássica Get Smart conta cartão grátis. ' * 40 + '</main>'
        return httpx.Response(200, text=html + '<script>' + script + '</script>',
                              headers={"content-type": "text/html"})

    mock_downloads(monkeypatch, handler)
    output = tmp_path / "chunks.json"
    ingestion.ingest(output, max_pages=3)
    matches = Retriever.from_file(output).retrieve(question)
    assert {str(chunk.source) for chunk in matches} == set(ingestion.SEED_URLS[:2])
    text = " ".join(chunk.text for chunk in matches)
    assert "Teclado físico" in text and "Teclado touch" in text
    assert "controle de estoque" in text


def test_english_pix_question_retrieves_portuguese_account_information():
    retriever = Retriever([
        Chunk(text="O Pix pode ser habilitado independentemente de qual banco estiver vinculado "
                   "para recebimento das vendas. Use o Aplicativo Getnet Brasil ou Portal Minha Conta.",
              source="https://site.getnet.com.br/pix/"),
        Chunk(text="A conta digital recebe vendas e permite receber Pix. Conta corrente e conta de pagamento.",
              source="https://site.getnet.com.br/get-ajuda-get-conta/o-que-e-a-get-conta/"),
        Chunk(text="Getnet is a global payment service for your business, restaurants and travel.",
              source="https://www.getnet.net/en/"),
    ])
    matches = retriever.retrieve("Do I need a bank account to receive my sales via Pix?", top_k=2)
    assert {str(chunk.source) for chunk in matches} == {
        "https://site.getnet.com.br/pix/",
        "https://site.getnet.com.br/get-ajuda-get-conta/o-que-e-a-get-conta/",
    }

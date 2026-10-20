import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.config import Settings
from app.schemas import AppError


class Gemini:
    """The only module allowed to call the Gemini API."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, prompt: str, *, schema: type[BaseModel] | None = None) -> str:
        key = self.settings.gemini_api_key.get_secret_value()
        if not key:
            raise AppError(503, "GEMINI_API_KEY não configurada.")
        config = types.GenerateContentConfig(temperature=0)
        if schema is not None:
            config.response_mime_type = "application/json"
            # Send standard JSON Schema directly; the legacy Schema conversion
            # can serialize extra="forbid" as an unsupported additional_properties field.
            config.response_json_schema = schema.model_json_schema()
        try:
            with genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=int(self.settings.gemini_timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            ) as client:
                response = client.models.generate_content(
                    model=self.settings.gemini_model, contents=prompt, config=config
                )
                text = response.text
        except httpx.TimeoutException as exc:
            raise AppError(504, "Tempo limite na chamada ao Gemini.") from exc
        except errors.APIError as exc:
            # Do not expose raw provider payloads, which may contain request data.
            if exc.code == 429:
                raise AppError(
                    429, "Limite de requisições ou cota do Gemini atingido (429). "
                    "Consulte os limites do projeto no Google AI Studio antes de tentar novamente."
                ) from exc
            if exc.code == 503:
                raise AppError(
                    503, "Gemini temporariamente indisponível (503), possivelmente por alta demanda. "
                    "Tente novamente mais tarde ou configure outro GEMINI_MODEL disponível."
                ) from exc
            if exc.code == 504:
                raise AppError(504, "Tempo limite na chamada ao Gemini (504).") from exc
            if exc.code in {401, 403}:
                detail = "Gemini recusou o acesso. Verifique GEMINI_API_KEY e as permissões do projeto."
            elif exc.code == 404:
                detail = "Modelo Gemini não encontrado ou não disponível. Verifique GEMINI_MODEL."
            elif exc.code == 400:
                detail = ("Gemini rejeitou a requisição (400). Verifique GEMINI_API_KEY, "
                          "GEMINI_MODEL e os parâmetros enviados.")
            else:
                detail = "Não foi possível obter resposta do Gemini."
            raise AppError(502, detail) from exc
        except httpx.HTTPError as exc:
            raise AppError(502, "Falha de conexão com o Gemini.") from exc
        if not text or not text.strip():
            raise AppError(502, "O Gemini retornou uma resposta vazia ou bloqueada.")
        return text.strip()

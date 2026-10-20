"""Optional local demo credentials. These are not production authentication."""

from hmac import compare_digest

from app.config import Settings
from app.schemas import AppError


def authorize_user(settings: Settings, user_id: str, api_key: str | None) -> None:
    if not settings.demo_api_keys:
        return  # Explicit unauthenticated local demo mode.
    identity = next((user for key, user in settings.demo_api_keys.items()
                     if compare_digest(key, api_key or "")), None)
    if identity is None:
        raise AppError(401, "Credencial de demonstração inválida ou ausente.")
    if identity != user_id:
        raise AppError(403, "Credencial não autorizada para este user_id.")


def authorize_reviewer(settings: Settings, api_key: str | None) -> None:
    key = settings.reviewer_api_key.get_secret_value()
    if key and not compare_digest(key, api_key or ""):
        raise AppError(401, "Credencial de revisão inválida ou ausente.")

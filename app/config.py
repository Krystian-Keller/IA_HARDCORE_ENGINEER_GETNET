import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = Field(default="gemini-flash-latest", min_length=1)
    gemini_timeout_seconds: float = Field(default=30, gt=0)
    http_timeout_seconds: float = Field(default=15, gt=0)
    data_dir: Path = ROOT / "data"
    history_max_messages: int = Field(default=12, ge=1, le=100)
    history_max_chars: int = Field(default=12000, ge=100, le=100000)
    demo_api_keys: dict[str, str] = Field(default_factory=dict, repr=False, exclude=True)
    reviewer_api_key: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def validate_identity(self) -> "Settings":
        if any(not key or not user for key, user in self.demo_api_keys.items()):
            raise ValueError("Demo identity mappings must be nonempty")
        if self.demo_api_keys and not self.reviewer_api_key.get_secret_value():
            raise ValueError("REVIEWER_API_KEY is required when DEMO_API_KEYS is configured")
        if self.reviewer_api_key.get_secret_value() in self.demo_api_keys:
            raise ValueError("Reviewer and customer credentials must differ")
        return self

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL") or "gemini-flash-latest",
            gemini_timeout_seconds=os.getenv("GEMINI_TIMEOUT_SECONDS", "30"),
            http_timeout_seconds=os.getenv("HTTP_TIMEOUT_SECONDS", "15"),
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data"))),
            history_max_messages=os.getenv("HISTORY_MAX_MESSAGES", "12"),
            history_max_chars=os.getenv("HISTORY_MAX_CHARS", "12000"),
            demo_api_keys=json.loads(os.getenv("DEMO_API_KEYS") or "{}"),
            reviewer_api_key=os.getenv("REVIEWER_API_KEY", ""),
        )

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = Field(default="gemini-flash-latest", min_length=1)
    gemini_timeout_seconds: float = Field(default=30, gt=0)
    http_timeout_seconds: float = Field(default=15, gt=0)
    data_dir: Path = ROOT / "data"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL") or "gemini-flash-latest",
            gemini_timeout_seconds=os.getenv("GEMINI_TIMEOUT_SECONDS", "30"),
            http_timeout_seconds=os.getenv("HTTP_TIMEOUT_SECONDS", "15"),
            data_dir=Path(os.getenv("DATA_DIR", str(ROOT / "data"))),
        )

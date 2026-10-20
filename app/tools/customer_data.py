import json
from pathlib import Path

from app.config import Settings
from app.schemas import AppError


def _get_customer(user_id: str, path: Path | None) -> dict:
    path = path if path is not None else Settings.from_env().data_dir / "customers.json"
    try:
        customers = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(customers, dict):
            raise ValueError("Invalid customer data")
        customer = customers.get(user_id)
        if customer is not None and not isinstance(customer, dict):
            raise ValueError("Invalid customer record")
    except (OSError, ValueError) as exc:
        raise AppError(503, "Base local de clientes indisponível.") from exc
    if customer is None:
        raise AppError(404, "Não há dados disponíveis para este user_id.")
    return customer


def get_customer_profile(user_id: str, path: Path | None = None) -> dict:
    customer = _get_customer(user_id, path)
    return {"user_id": user_id, "data_is_fictional": True,
            "profile": customer.get("profile", {}),
            "receiving_configuration": customer.get("receiving_configuration", {})}


def get_customer_activity(user_id: str, path: Path | None = None) -> dict:
    customer = _get_customer(user_id, path)
    return {"user_id": user_id, "data_is_fictional": True,
            "snapshot_at": customer.get("snapshot_at"),
            "activity": customer.get("activity", {})}

import pytest

from app.schemas import AppError
from app.tools.customer_data import get_customer_activity, get_customer_profile


def test_existing_customer_profile(settings):
    profile = get_customer_profile("cliente1988", settings.data_dir / "customers.json")
    assert profile["profile"]["business_name"] == "Café Demonstração"
    assert profile["receiving_configuration"]["pix_enabled"] is True
    assert "activity" not in profile


def test_customer_activity(settings):
    result = get_customer_activity("cliente1988", settings.data_dir / "customers.json")
    assert result["data_is_fictional"] is True
    assert result["snapshot_at"] == "2026-10-15T09:00:00-03:00"
    activity = result["activity"]
    assert len(activity["recent_sales"]) == 2
    assert activity["expected_deposits"][0]["amount"] == 147.75
    assert activity["device"]["status"] == "offline"
    assert activity["last_transaction_error"]["code"] == "DEMO_DECLINED"


@pytest.mark.parametrize("tool", [get_customer_profile, get_customer_activity])
def test_unknown_customer(tool, settings):
    with pytest.raises(AppError, match="Não há dados") as error:
        tool("unknown", settings.data_dir / "customers.json")
    assert error.value.status_code == 404


@pytest.mark.parametrize("content", [None, "broken", "[]", '{"cliente1988":42}'])
def test_unavailable_customer_data(tmp_path, content):
    path = tmp_path / "customers.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(AppError) as error:
        get_customer_activity("cliente1988", path)
    assert error.value.status_code == 503

from core.db.models import CorporateAction, CorporateActionLedger


def test_corporate_action_schema_fields() -> None:
    fields = set(CorporateAction.model_fields)
    assert {"symbol", "action_type", "ex_date", "record_date", "pay_date", "public_date", "params_json", "source", "raw_json"}.issubset(fields)


def test_ca_ledger_schema_fields() -> None:
    fields = set(CorporateActionLedger.model_fields)
    assert {"portfolio_id", "symbol", "ex_date", "action_type", "qty_before", "qty_after", "cash_delta", "avg_cost_before", "avg_cost_after", "fee", "tax", "notes_json"}.issubset(fields)

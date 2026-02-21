from core.eval_lab.consistency import check_end_identity


def test_identity_gross_net_cost_tolerance() -> None:
    gross_end = 1.234567
    net_end = 1.123456
    total_cost = gross_end - net_end
    assert check_end_identity(gross_end, net_end, total_cost)

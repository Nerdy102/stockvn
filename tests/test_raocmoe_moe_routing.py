from core.raocmoe.routing import HedgeRouter


def test_routing_weight_increases_for_better_expert() -> None:
    r = HedgeRouter(["SIDEWAYS"], ["A", "B"], eta=0.25, loss_clip=5.0)
    for _ in range(80):
        r.update("SIDEWAYS", {"A": 0.02, "B": -0.01}, realized_r=0.01, vol=0.02)
    assert r.weights["SIDEWAYS"]["A"] > r.weights["SIDEWAYS"]["B"]

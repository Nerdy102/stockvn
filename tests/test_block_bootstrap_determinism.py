from core.eval_lab.bootstrap import block_bootstrap_samples


def test_block_bootstrap_determinism() -> None:
    vals = [float(i) / 100.0 for i in range(100)]
    a = block_bootstrap_samples(vals, n_samples=50, block_size=10, seed=42)
    b = block_bootstrap_samples(vals, n_samples=50, block_size=10, seed=42)
    assert (a == b).all()

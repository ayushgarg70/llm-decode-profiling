from bench.fixed_cost import SHAPES, DECODE_ROLE, fit_line, pool_size


def test_fit_line_recovers_intercept_and_bandwidth():
    bw = 6.6e12
    xs = [2e6, 8e6, 33.6e6, 117e6, 1.05e9]
    ys = [3.0 + x / bw * 1e6 for x in xs]
    a, b = fit_line(xs, ys)
    assert abs(a - 3.0) < 1e-6 and abs(b / bw - 1) < 1e-9


def test_pool_is_larger_than_l2_and_has_four_copies():
    assert pool_size(1024 * 4096 * 2, 2 * 2**30) * 1024 * 4096 * 2 >= 2 * 2**30
    assert pool_size(128256 * 4096 * 2, 2 * 2**30) == 4


def test_every_decode_shape_is_swept():
    assert set(DECODE_ROLE) <= set(SHAPES)


def test_fixed_cost_section_renders(tmp_path, monkeypatch):
    import csv

    import analysis.results_table as rt

    (tmp_path / "data").mkdir()
    fields = ["sweep_id", "gpu", "roof_Bps", "fit_fixed_us", "fit_Bps", "pool_gib", "N", "K", "weight_bytes",
              "decode_role", "kernels", "us_per_kernel", "TBps", "above_roof_us"]
    with (tmp_path / "data/fixed_cost.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"sweep_id": "x", "gpu": "NVIDIA B200", "roof_Bps": 6.6e12, "fit_fixed_us": 3.1, "fit_Bps": 6.5e12,
                    "pool_gib": 2.0, "N": 4096, "K": 4096, "weight_bytes": 33554432, "decode_role": "q, o",
                    "kernels": 512, "us_per_kernel": 8.2, "TBps": 4.1, "above_roof_us": 3.1})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rt, "GPU", "B200")
    out = rt.fixed_cost_section()
    assert "fixed 3.10 µs per kernel" in out and "| 4096 | 4096 | 33.6 | q, o |" in out

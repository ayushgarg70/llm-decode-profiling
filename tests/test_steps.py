"""data/steps/ (per-step kernel lists extracted from the nsys traces) is what the timeline charts and the
per-kernel tables in notes/RESULTS.md are built from; check it's present, well-formed and consistent."""

import csv
from pathlib import Path

import pytest

from analysis.group_kernels import classify
from analysis.plots import _step_kernels

STEPS = sorted(Path("data/steps").glob("*.csv"))


def test_steps_exist_for_the_charts():
    for trace in ("E1_own_eager_light", "E3_eager_graph", "E3_compiled", "E3_compiled_graph"):
        assert Path(f"data/steps/{trace}_step2.csv").exists()


@pytest.mark.parametrize("path", STEPS, ids=lambda p: p.stem)
def test_step_csv_well_formed(path):
    rows = list(csv.DictReader(path.open()))
    assert rows and float(rows[0]["start_us"]) == 0.0
    starts = [float(r["start_us"]) for r in rows]
    assert starts == sorted(starts)
    for r in rows:
        assert float(r["dur_us"]) >= 0 and int(r["grid"]) > 0
        assert r["group"] == classify(r["kernel"])


def test_eager_step_matches_the_report():
    ks = _step_kernels("E1_own_eager_light")
    assert len(ks) == 1583  # kernels per eager step (E2)


def test_h100_vs_b200_roles():
    """Matmul roles are assigned by position: every layer has q, k, v, o, gate, up, down, then lm_head once."""
    from analysis import results_table as rt

    rows = {r["key"]: r for r in rt.compare_rows()}
    for gpu in ("H100", "B200"):
        # the biggest matmul is lm_head and the MLP matmuls are ~4x the k/v projections' bytes: check the order holds
        assert rows["lm_head"][f"{gpu}_us"] > rows["mlp"][f"{gpu}_us"] > rows["q"][f"{gpu}_us"] > rows["kv"][f"{gpu}_us"]
        assert rows["mlp"][f"{gpu}_extra"] > 0

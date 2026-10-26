"""T0.6: the plot script reads two dummy CSVs of the spec schema and writes one PNG
with a mean line and a std band, and it refuses a CSV with the wrong header.
"""

import numpy as np
import pytest

from plots.make_plots import aggregate, make_plots, plot_runs


def _write_csv(path, returns):
    lines = ["step,eval_return,wall_time_sec"]
    for i, value in enumerate(returns, start=1):
        lines.append(f"{i * 10},{value},{i * 0.01}")
    path.write_text("\n".join(lines) + "\n")


def test_aggregate_two_seeds(tmp_path):
    a = tmp_path / "seed0.csv"
    b = tmp_path / "seed1.csv"
    _write_csv(a, [0.0, 2.0, 4.0])
    _write_csv(b, [2.0, 4.0, 6.0])
    steps, mean, std = aggregate([str(a), str(b)])
    np.testing.assert_array_equal(steps, [10, 20, 30])
    np.testing.assert_allclose(mean, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(std, [1.0, 1.0, 1.0])


def test_plot_runs_writes_png(tmp_path):
    a = tmp_path / "seed0.csv"
    b = tmp_path / "seed1.csv"
    _write_csv(a, [-10.0, -5.0, -2.0])
    _write_csv(b, [-12.0, -6.0, -3.0])
    out = tmp_path / "curve.png"
    plot_runs([str(a), str(b)], str(out), label="demo")
    assert out.exists() and out.stat().st_size > 0


def test_make_plots_over_results_tree(tmp_path):
    run_dir = tmp_path / "results" / "bandit" / "copdac_q"
    run_dir.mkdir(parents=True)
    _write_csv(run_dir / "seed0.csv", [5.0, 3.0, 1.0])
    _write_csv(run_dir / "seed1.csv", [6.0, 4.0, 2.0])
    out_dir = tmp_path / "out"
    written = make_plots(str(tmp_path / "results"), str(out_dir))
    assert len(written) == 1
    assert (out_dir / "bandit_learning_curve.png").exists()


def test_plot_rejects_bad_header(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("iteration,value,seconds\n1,0.0,0.0\n")
    with pytest.raises(ValueError):
        plot_runs([str(bad)], str(tmp_path / "out.png"))

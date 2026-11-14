"""T0.6: the plot script reads two dummy CSVs of the spec schema and writes one PNG
with a mean line and a std band, and it refuses a CSV with the wrong header.
"""

import numpy as np
import pytest

from plots.make_plots import _axis_is_log, aggregate, make_plots, plot_runs, summarize_runs


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
    assert (out_dir / "bandit_learning_curve.png").exists()
    assert (out_dir / "results_table.md").exists()
    assert str(out_dir / "bandit_learning_curve.png") in written
    assert str(out_dir / "results_table.md") in written


def test_make_plots_ignores_episode_logs(tmp_path):
    run_dir = tmp_path / "results" / "lunarlander" / "ddpg"
    run_dir.mkdir(parents=True)
    _write_csv(run_dir / "seed0.csv", [1.0, 2.0])
    (run_dir / "seed0_episodes.csv").write_text("episode,step,reward\n1,10,0.0\n")
    written = make_plots(str(tmp_path / "results"), str(tmp_path / "out"))
    pngs = [path for path in written if path.endswith(".png")]
    assert pngs == [str(tmp_path / "out" / "lunarlander_learning_curve.png")]
    table = (tmp_path / "out" / "results_table.md").read_text()
    assert "| lunarlander | ddpg | 0 | 1 |" in table


def test_results_table_mean_wallclock_and_threshold(tmp_path):
    lunar = tmp_path / "results" / "lunarlander" / "td3"
    bandit = tmp_path / "results" / "bandit_m10" / "copdac_q"
    lunar.mkdir(parents=True)
    bandit.mkdir(parents=True)
    (lunar / "seed0.csv").write_text(
        "step,eval_return,wall_time_sec\n10,0,1\n20,251,5\n"
    )
    (lunar / "seed4.csv").write_text(
        "step,eval_return,wall_time_sec\n10,0,3\n20,151,7\n"
    )
    (lunar / "seed0_episodes.csv").write_text("episode,step,reward\n1,10,0\n")
    _write_csv(bandit / "seed0.csv", [4.0, 1.0])
    out_dir = tmp_path / "out"
    make_plots(str(tmp_path / "results"), str(out_dir), log_y=True)
    table = (out_dir / "results_table.md").read_text()
    # mean return 201 exceeds 200 at step 20; seeds 1, 2, 3 are a gap, not zeros.
    assert "| lunarlander | td3 | 0,4 | 2 | 201 | 50 | 6 | 1 | 20 |" in table
    assert "| bandit_m10 | copdac_q | 0 | 1 | 1 | 0 |" in table
    assert "| bandit_m10 | copdac_q | 0 | 1 | 1 | 0 | 0.02 | 0 | not defined |" in table
    assert "seed0_episodes" not in table
    assert (out_dir / "lunarlander_learning_curve.png").exists()
    assert (out_dir / "bandit_m10_learning_curve.png").exists()


def test_log_axis_is_only_for_the_bandit():
    assert _axis_is_log("bandit_m10", True)
    assert not _axis_is_log("lunarlander", True)
    assert not _axis_is_log("bandit_m50", False)


def test_threshold_stays_never_when_the_mean_does_not_exceed_200(tmp_path):
    lunar = tmp_path / "results" / "lunarlander" / "ddpg"
    lunar.mkdir(parents=True)
    (lunar / "seed0.csv").write_text("step,eval_return,wall_time_sec\n10,100,1\n20,200,2\n")
    row = summarize_runs("lunarlander", "ddpg", [str(lunar / "seed0.csv")])
    assert row["threshold_step"] == "never"
    assert row["final_mean"] == "200"


def test_plot_rejects_bad_header(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("iteration,value,seconds\n1,0.0,0.0\n")
    with pytest.raises(ValueError):
        plot_runs([str(bad)], str(tmp_path / "out.png"))

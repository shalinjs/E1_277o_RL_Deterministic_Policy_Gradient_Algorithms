"""T0.2: every module imports with no training, and run.py exposes five arguments."""

import importlib

import pytest

MODULES = [
    "envs.bandit",
    "features.rbf",
    "common.seeding",
    "common.logging",
    "common.wrappers",
    "common.runner",
    "agents.stochastic_ac",
    "agents.offpac",
    "agents.copdac_q",
    "agents.ddpg_td3",
    "plots.make_plots",
    "run",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_run_parser_has_five_arguments():
    import run

    parser = run.build_parser()
    dests = {action.dest for action in parser._actions if action.dest != "help"}
    assert dests == {"agent", "env", "seed", "steps", "config"}


def test_run_help_does_not_train(capsys):
    import run

    with pytest.raises(SystemExit) as exc:
        run.main(["--help"])
    assert exc.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_make_agent_defers_to_phase_1():
    import run

    with pytest.raises(NotImplementedError):
        run.make_agent("copdac_q", env=None, config={})

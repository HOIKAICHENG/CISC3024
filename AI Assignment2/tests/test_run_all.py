import builtins
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_run_all_help():
    result = subprocess.run(
        [sys.executable, "-m", "code.run_all", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--quick" in result.stdout
    assert "--skip-report" in result.stdout


@pytest.mark.parametrize(
    ("quick", "checkpoint_name"),
    [(True, "map_checkpoint_quick.pt"), (False, "map_checkpoint.pt")],
)
def test_run_experiment_propagates_one_mode_to_every_stage(
    tmp_path, monkeypatch, quick, checkpoint_name
):
    from code import run_all

    config = SimpleNamespace(results_dir=str(tmp_path))
    loaders = object()
    map_model = object()
    members = [object()]
    calls = []

    monkeypatch.setattr(run_all, "ExperimentConfig", lambda: config)

    def build_loaders(config_arg, *, quick):
        calls.append(("data", config_arg, quick))
        return loaders

    def run_map_training(config_arg, loaders_arg, *, quick):
        calls.append(("map", config_arg, loaders_arg, quick))
        return map_model, [{"epoch": 1}]

    def run_abnn_training(
        config_arg, loaders_arg, checkpoint_arg, *, quick
    ):
        calls.append(
            (
                "abnn",
                config_arg,
                loaders_arg,
                checkpoint_arg,
                quick,
            )
        )
        return members, [[{"epoch": 1}]]

    def evaluate_methods(
        config_arg, loaders_arg, map_model_arg, members_arg, *, quick
    ):
        calls.append(
            (
                "evaluate",
                config_arg,
                loaders_arg,
                map_model_arg,
                members_arg,
                quick,
            )
        )

    def make_all_figures(results_dir, figures_dir, *, quick):
        calls.append(("figures", results_dir, figures_dir, quick))

    monkeypatch.setattr(run_all, "build_loaders", build_loaders)
    monkeypatch.setattr(run_all, "run_map_training", run_map_training)
    monkeypatch.setattr(run_all, "run_abnn_training", run_abnn_training)
    monkeypatch.setattr(run_all, "evaluate_methods", evaluate_methods)
    monkeypatch.setattr(run_all, "make_all_figures", make_all_figures)

    original_import = builtins.__import__

    def reject_report_import(name, *args, **kwargs):
        if name == "build_report":
            pytest.fail("--skip-report imported build_report")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_report_import)

    run_all.run_experiment(quick=quick, skip_report=True)

    assert calls == [
        ("data", config, quick),
        ("map", config, loaders, quick),
        (
            "abnn",
            config,
            loaders,
            Path(tmp_path, checkpoint_name),
            quick,
        ),
        ("evaluate", config, loaders, map_model, members, quick),
        ("figures", str(tmp_path), "figures", quick),
    ]


def test_report_is_imported_and_invoked_only_when_requested(
    tmp_path, monkeypatch
):
    from code import run_all

    config = SimpleNamespace(results_dir=str(tmp_path))
    monkeypatch.setattr(run_all, "ExperimentConfig", lambda: config)
    monkeypatch.setattr(
        run_all,
        "build_loaders",
        lambda config, *, quick: object(),
    )
    monkeypatch.setattr(
        run_all,
        "run_map_training",
        lambda config, loaders, *, quick: (object(), []),
    )
    monkeypatch.setattr(
        run_all,
        "run_abnn_training",
        lambda config, loaders, checkpoint, *, quick: ([object()], []),
    )
    monkeypatch.setattr(
        run_all,
        "evaluate_methods",
        lambda config, loaders, map_model, members, *, quick: {},
    )
    monkeypatch.setattr(
        run_all,
        "make_all_figures",
        lambda results_dir, figures_dir, *, quick: [],
    )
    report_calls = []
    report_module = ModuleType("build_report")
    report_module.build_report = lambda *, quick: report_calls.append(quick)
    monkeypatch.setitem(sys.modules, "build_report", report_module)

    run_all.run_experiment(quick=True, skip_report=False)

    assert report_calls == [True]


def test_main_passes_cli_policy_to_orchestrator(monkeypatch):
    from code import run_all

    calls = []
    monkeypatch.setattr(
        run_all,
        "run_experiment",
        lambda *, quick, skip_report: calls.append((quick, skip_report)),
    )
    monkeypatch.setattr(
        sys, "argv", ["code.run_all", "--quick", "--skip-report"]
    )

    run_all.main()

    assert calls == [(True, True)]

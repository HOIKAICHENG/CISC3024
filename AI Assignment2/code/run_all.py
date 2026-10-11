import argparse
from pathlib import Path

from .config import ExperimentConfig
from .data import build_loaders
from .evaluate import evaluate_methods
from .make_figures import make_all_figures
from .train_abnn import run_abnn_training
from .train_map import run_map_training


def run_experiment(*, quick: bool = False, skip_report: bool = False) -> None:
    config = ExperimentConfig()
    loaders = build_loaders(config, quick=quick)
    map_model, _ = run_map_training(config, loaders, quick=quick)
    suffix = "_quick" if quick else ""
    map_checkpoint = (
        Path(config.results_dir) / f"map_checkpoint{suffix}.pt"
    )
    members, _ = run_abnn_training(
        config,
        loaders,
        map_checkpoint,
        quick=quick,
    )
    evaluate_methods(
        config,
        loaders,
        map_model,
        members,
        quick=quick,
    )
    make_all_figures(config.results_dir, "figures", quick=quick)
    if not skip_report:
        from build_report import build_report

        build_report(quick=quick)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ABNN assignment experiment"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the small end-to-end smoke configuration",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="do not invoke build_report.py",
    )
    args = parser.parse_args()
    run_experiment(quick=args.quick, skip_report=args.skip_report)


if __name__ == "__main__":
    main()

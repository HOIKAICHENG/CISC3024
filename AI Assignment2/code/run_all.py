import argparse
import hashlib
import json
from pathlib import Path

from .config import ExperimentConfig, run_metadata
from .data import build_loaders
from .evaluate import evaluate_methods
from .make_figures import (
    figures_are_current,
    load_valid_evaluation,
    make_all_figures,
)
from .train_abnn import run_abnn_training
from .train_map import run_map_training


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(map_checkpoint, member_checkpoints) -> dict:
    map_checkpoint = Path(map_checkpoint)
    member_checkpoints = [Path(path) for path in member_checkpoints]
    return {
        "map_checkpoint": {
            "path": map_checkpoint.name,
            "sha256": _sha256(map_checkpoint),
        },
        "member_checkpoints": [
            {
                "path": checkpoint.name,
                "sha256": _sha256(checkpoint),
            }
            for checkpoint in member_checkpoints
        ],
    }


def evaluation_is_current(
    path,
    *,
    config: ExperimentConfig,
    quick: bool,
    provenance: dict,
) -> bool:
    try:
        evaluation = load_valid_evaluation(path)
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    return (
        evaluation.get("config")
        == json.loads(json.dumps(config.to_dict()))
        and evaluation.get("run") == run_metadata(config, quick)
        and evaluation.get("provenance") == provenance
    )


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
    member_checkpoints = [
        Path(config.results_dir) / f"abnn_member_{index}{suffix}.pt"
        for index in range(len(members))
    ]
    provenance = checkpoint_provenance(
        map_checkpoint,
        member_checkpoints,
    )
    evaluation_path = (
        Path(config.results_dir) / f"evaluation{suffix}.json"
    )
    if not evaluation_is_current(
        evaluation_path,
        config=config,
        quick=quick,
        provenance=provenance,
    ):
        evaluate_methods(
            config,
            loaders,
            map_model,
            members,
            quick=quick,
            provenance=provenance,
        )
    if not figures_are_current(
        config.results_dir,
        "figures",
        quick=quick,
    ):
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

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from code.make_figures import (
    _plot_ood_entropy,
    _reliability_points,
    _uncertainty_records,
    make_all_figures,
)


METHODS = ("map", "abnn_single", "abnn_full")


def _fixture():
    probabilities = {
        "map": [
            [0.80, 0.15, 0.05],
            [0.30, 0.60, 0.10],
            [0.15, 0.70, 0.15],
            [0.40, 0.20, 0.40],
        ],
        "abnn_single": [
            [0.75, 0.20, 0.05],
            [0.25, 0.65, 0.10],
            [0.15, 0.55, 0.30],
            [0.30, 0.25, 0.45],
        ],
        "abnn_full": [
            [0.90, 0.07, 0.03],
            [0.20, 0.70, 0.10],
            [0.20, 0.50, 0.30],
            [0.25, 0.30, 0.45],
        ],
    }
    targets = [0, 1, 2, 2]
    entropies = {
        name: (
            -np.sum(
                np.asarray(values) * np.log(np.asarray(values)),
                axis=1,
            )
        ).tolist()
        for name, values in probabilities.items()
    }
    methods = {}
    for name in METHODS:
        methods[name] = {
            "classification": {
                "accuracy": 0.75,
                "nll": 0.7,
                "brier": 0.4,
                # Deliberately unrelated to the probabilities. The
                # reliability diagram must not use this summary value.
                "ece": 0.99,
            },
            "ood": {
                "svhn": {"auroc": 0.75, "aupr": 0.72, "fpr95": 0.50},
                "cifar100": {
                    "auroc": 0.65,
                    "aupr": 0.62,
                    "fpr95": 0.75,
                },
            },
            "ood_predictive_entropy": {
                "svhn": [0.92, 0.88, 0.81, 0.79],
                "cifar100": [0.74, 0.70, 0.68, 0.65],
            },
            "probabilities": probabilities[name],
            "targets": targets,
            "predictive_entropy": entropies[name],
        }
    images = []
    for index in range(len(targets)):
        image = np.zeros((3, 8, 8), dtype=float)
        image[0] = index / 4
        image[1] = (index + 1) / 5
        image[2] = (index + 2) / 6
        images.append(image.tolist())
    return {
        "methods": methods,
        "example_images": images,
        "example_targets": targets,
    }


def _write_artifacts(results, quick):
    suffix = "_quick" if quick else ""
    evaluation = _fixture()
    (results / f"evaluation{suffix}.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    (results / f"map_history{suffix}.json").write_text(
        json.dumps(
            {
                "history": [
                    {"epoch": 1, "loss": 1.2, "accuracy": 0.45},
                    {"epoch": 2, "loss": 0.8, "accuracy": 0.70},
                ]
            }
        ),
        encoding="utf-8",
    )
    (results / f"abnn_history{suffix}.json").write_text(
        json.dumps(
            {
                "members": [
                    [
                        {"epoch": 1, "loss": 1.0},
                        {"epoch": 2, "loss": 0.7},
                    ],
                    [
                        {"epoch": 1, "loss": 1.1},
                        {"epoch": 2, "loss": 0.75},
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    return evaluation


@pytest.mark.parametrize("quick", [True, False])
def test_figures_are_nonempty_pngs_for_quick_and_full_artifacts(
    tmp_path, quick
):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    _write_artifacts(results, quick)

    paths = make_all_figures(results, figures, quick=quick)

    assert [path.name for path in paths] == [
        "architecture.png",
        "training_curves.png",
        "reliability.png",
        "ood_entropy.png",
        "metrics_comparison.png",
        "uncertainty_examples.png",
    ]
    for path in paths:
        assert path.parent == figures
        assert path.stat().st_size > 1_000
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 600
            assert image.height >= 400


def test_reliability_is_recomputed_from_saved_probabilities():
    probabilities = np.asarray(
        [[0.90, 0.10], [0.75, 0.25], [0.35, 0.65]]
    )
    targets = np.asarray([0, 1, 1])

    confidence, accuracy = _reliability_points(
        probabilities, targets, bins=5
    )

    assert confidence == pytest.approx([0.7, 0.9])
    assert accuracy == pytest.approx([0.5, 1.0])


def test_entropy_plot_names_id_and_both_ood_datasets():
    figure = _plot_ood_entropy(_fixture())

    labels = figure.axes[0].get_legend_handles_labels()[1]
    plt.close(figure)

    assert labels == ["CIFAR-10 (ID)", "SVHN (OOD)", "CIFAR-100 (OOD)"]


def test_uncertainty_records_keep_examples_aligned_when_sorted():
    evaluation = _fixture()
    method = evaluation["methods"]["abnn_full"]
    expected_order = np.argsort(method["predictive_entropy"])

    records = _uncertainty_records(evaluation)

    assert [record["target"] for record in records] == [
        evaluation["example_targets"][index] for index in expected_order
    ]
    assert [record["prediction"] for record in records] == [
        int(np.argmax(method["probabilities"][index]))
        for index in expected_order
    ]
    assert [record["entropy"] for record in records] == pytest.approx(
        [method["predictive_entropy"][index] for index in expected_order]
    )
    for record, index in zip(records, expected_order):
        np.testing.assert_allclose(
            record["image"],
            evaluation["example_images"][index],
        )


def test_schema_errors_identify_the_artifact_field(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    evaluation = _write_artifacts(results, quick=True)
    invalid = copy.deepcopy(evaluation)
    del invalid["methods"]["abnn_full"]["probabilities"]
    (results / "evaluation_quick.json").write_text(
        json.dumps(invalid), encoding="utf-8"
    )

    with pytest.raises(
        ValueError,
        match=r"evaluation_quick\.json.*methods\.abnn_full\.probabilities",
    ):
        make_all_figures(results, tmp_path / "figures", quick=True)


def test_schema_allows_multiclass_nll_and_brier_above_one(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    evaluation = _write_artifacts(results, quick=True)
    for method in evaluation["methods"].values():
        method["classification"]["nll"] = 2.3
        method["classification"]["brier"] = 1.2
    (results / "evaluation_quick.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )

    paths = make_all_figures(
        results,
        tmp_path / "figures",
        quick=True,
    )

    assert len(paths) == 6


def test_missing_artifact_error_names_expected_file(tmp_path):
    with pytest.raises(
        FileNotFoundError, match=r"evaluation_quick\.json"
    ):
        make_all_figures(
            tmp_path / "results",
            tmp_path / "figures",
            quick=True,
        )

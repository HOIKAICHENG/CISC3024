"""Generate report figures exclusively from persisted experiment artifacts."""

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

from .engine import save_json


METHODS = ("map", "abnn_single", "abnn_full")
METHOD_LABELS = {
    "map": "MAP",
    "abnn_single": "ABNN (single)",
    "abnn_full": "ABNN (ensemble)",
}
METHOD_COLORS = {
    "map": "#4C78A8",
    "abnn_single": "#F58518",
    "abnn_full": "#54A24B",
}
OOD_DATASETS = ("svhn", "cifar100")
OOD_LABELS = {"svhn": "SVHN", "cifar100": "CIFAR-100"}
OUTPUT_NAMES = (
    "architecture.png",
    "training_curves.png",
    "reliability.png",
    "ood_entropy.png",
    "metrics_comparison.png",
    "uncertainty_examples.png",
)
MANIFEST_NAME = "manifest.json"
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    finally:
        plt.close(fig)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_png(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{path.name}: expected a non-empty PNG")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise ValueError(f"{path.name}: expected a valid PNG")
    except (OSError, SyntaxError) as error:
        raise ValueError(f"{path.name}: expected a valid PNG") from error


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required figure artifact not found: {path}. "
            "Run the matching quick/full experiment first."
        )
    try:
        with path.open(encoding="utf-8") as artifact:
            return json.load(artifact)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path.name}: invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error


def _schema_error(path: Path, field: str, detail: str):
    raise ValueError(f"{path.name}: invalid field {field}: {detail}")


def _mapping(value, path: Path, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        _schema_error(path, field, "expected an object")
    return value


def _sequence(value, path: Path, field: str) -> Sequence:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        _schema_error(path, field, "expected an array")
    return value


def _required(container: Mapping, key: str, path: Path, field: str):
    full_field = f"{field}.{key}" if field else key
    if key not in container:
        raise ValueError(
            f"{path.name}: missing required field {full_field}"
        )
    return container[key]


def _validate_artifact_mode(payload, path: Path, quick: bool) -> None:
    root = _mapping(payload, path, "root")
    run = _mapping(
        _required(root, "run", path, ""),
        path,
        "run",
    )
    expected_mode = "quick" if quick else "full"
    if run.get("mode") != expected_mode or run.get("quick") is not quick:
        _schema_error(
            path,
            "run",
            f"expected {expected_mode} mode metadata",
        )


def _finite_number(value, path: Path, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _schema_error(path, field, "expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        _schema_error(path, field, "expected a finite number")
    return number


def _number_array(
    value,
    path: Path,
    field: str,
    *,
    dimensions: int,
    nonempty: bool = True,
) -> np.ndarray:
    _sequence(value, path, field)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        _schema_error(path, field, f"expected a regular numeric array ({error})")
    if array.ndim != dimensions:
        _schema_error(
            path,
            field,
            f"expected {dimensions} dimensions, got {array.ndim}",
        )
    if nonempty and (array.size == 0 or any(size == 0 for size in array.shape)):
        _schema_error(path, field, "must not be empty")
    if not np.isfinite(array).all():
        _schema_error(path, field, "must contain only finite numbers")
    return array


def _validate_metrics(
    value,
    path: Path,
    field: str,
    required_keys: Sequence[str],
) -> None:
    metrics = _mapping(value, path, field)
    for key in required_keys:
        metric = _finite_number(
            _required(metrics, key, path, field),
            path,
            f"{field}.{key}",
        )
        if metric < 0.0:
            _schema_error(
                path,
                f"{field}.{key}",
                "expected a non-negative value",
            )
        if key not in ("nll", "brier") and metric > 1.0:
            _schema_error(
                path,
                f"{field}.{key}",
                "expected a value between 0 and 1",
            )


def _validate_evaluation(evaluation, path: Path) -> None:
    root = _mapping(evaluation, path, "root")
    methods = _mapping(
        _required(root, "methods", path, ""),
        path,
        "methods",
    )
    reference_targets = None
    class_count = None

    for method_name in METHODS:
        method_field = f"methods.{method_name}"
        method = _mapping(
            _required(methods, method_name, path, "methods"),
            path,
            method_field,
        )
        _validate_metrics(
            _required(method, "classification", path, method_field),
            path,
            f"{method_field}.classification",
            ("accuracy", "nll", "brier", "ece"),
        )

        ood = _mapping(
            _required(method, "ood", path, method_field),
            path,
            f"{method_field}.ood",
        )
        ood_entropy = _mapping(
            _required(
                method,
                "ood_predictive_entropy",
                path,
                method_field,
            ),
            path,
            f"{method_field}.ood_predictive_entropy",
        )
        for dataset in OOD_DATASETS:
            _validate_metrics(
                _required(
                    ood,
                    dataset,
                    path,
                    f"{method_field}.ood",
                ),
                path,
                f"{method_field}.ood.{dataset}",
                ("auroc", "aupr", "fpr95"),
            )
            entropy = _number_array(
                _required(
                    ood_entropy,
                    dataset,
                    path,
                    f"{method_field}.ood_predictive_entropy",
                ),
                path,
                (
                    f"{method_field}.ood_predictive_entropy."
                    f"{dataset}"
                ),
                dimensions=1,
            )
            if np.any(entropy < 0):
                _schema_error(
                    path,
                    (
                        f"{method_field}.ood_predictive_entropy."
                        f"{dataset}"
                    ),
                    "entropy values must be non-negative",
                )

        probabilities = _number_array(
            _required(method, "probabilities", path, method_field),
            path,
            f"{method_field}.probabilities",
            dimensions=2,
        )
        if probabilities.shape[1] < 2:
            _schema_error(
                path,
                f"{method_field}.probabilities",
                "expected probabilities for at least two classes",
            )
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            _schema_error(
                path,
                f"{method_field}.probabilities",
                "values must be between 0 and 1",
            )
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            _schema_error(
                path,
                f"{method_field}.probabilities",
                "each row must sum to 1",
            )

        raw_targets = _sequence(
            _required(method, "targets", path, method_field),
            path,
            f"{method_field}.targets",
        )
        if not raw_targets:
            _schema_error(
                path,
                f"{method_field}.targets",
                "must not be empty",
            )
        if any(
            isinstance(target, bool) or not isinstance(target, int)
            for target in raw_targets
        ):
            _schema_error(
                path,
                f"{method_field}.targets",
                "expected integer class indices",
            )
        targets = np.asarray(raw_targets, dtype=int)
        if len(targets) != len(probabilities):
            _schema_error(
                path,
                f"{method_field}.targets",
                "length must match probabilities",
            )
        if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
            _schema_error(
                path,
                f"{method_field}.targets",
                "contains a class outside the probability columns",
            )

        entropy = _number_array(
            _required(
                method,
                "predictive_entropy",
                path,
                method_field,
            ),
            path,
            f"{method_field}.predictive_entropy",
            dimensions=1,
        )
        if len(entropy) != len(probabilities):
            _schema_error(
                path,
                f"{method_field}.predictive_entropy",
                "length must match probabilities",
            )
        if np.any(entropy < 0):
            _schema_error(
                path,
                f"{method_field}.predictive_entropy",
                "entropy values must be non-negative",
            )

        if reference_targets is None:
            reference_targets = targets
            class_count = probabilities.shape[1]
        else:
            if not np.array_equal(targets, reference_targets):
                _schema_error(
                    path,
                    f"{method_field}.targets",
                    "must use the same sample order as MAP",
                )
            if probabilities.shape[1] != class_count:
                _schema_error(
                    path,
                    f"{method_field}.probabilities",
                    "class count must match MAP",
                )

    images = _number_array(
        _required(root, "example_images", path, ""),
        path,
        "example_images",
        dimensions=4,
    )
    if images.shape[1] not in (1, 3, 4):
        _schema_error(
            path,
            "example_images",
            "expected NCHW images with 1, 3, or 4 channels",
        )
    if np.any(images < 0) or np.any(images > 1):
        _schema_error(
            path,
            "example_images",
            "de-normalized pixel values must be between 0 and 1",
        )
    raw_example_targets = _sequence(
        _required(root, "example_targets", path, ""),
        path,
        "example_targets",
    )
    if any(
        isinstance(target, bool) or not isinstance(target, int)
        for target in raw_example_targets
    ):
        _schema_error(
            path,
            "example_targets",
            "expected integer class indices",
        )
    example_targets = np.asarray(raw_example_targets, dtype=int)
    if len(example_targets) != len(images):
        _schema_error(
            path,
            "example_targets",
            "length must match example_images",
        )
    if len(images) > len(reference_targets):
        _schema_error(
            path,
            "example_images",
            "cannot outnumber saved predictions",
        )
    if not np.array_equal(
        example_targets,
        reference_targets[: len(example_targets)],
    ):
        _schema_error(
            path,
            "example_targets",
            "must match the first saved prediction targets",
        )


def load_valid_evaluation(path) -> dict:
    path = Path(path)
    evaluation = _read_json(path)
    _validate_evaluation(evaluation, path)
    return evaluation


def _validate_history_rows(rows, path: Path, field: str) -> None:
    rows = _sequence(rows, path, field)
    if not rows:
        _schema_error(path, field, "must not be empty")
    for index, row_value in enumerate(rows):
        row_field = f"{field}[{index}]"
        row = _mapping(row_value, path, row_field)
        epoch = _finite_number(
            _required(row, "epoch", path, row_field),
            path,
            f"{row_field}.epoch",
        )
        if epoch <= 0 or not epoch.is_integer():
            _schema_error(
                path,
                f"{row_field}.epoch",
                "expected a positive integer",
            )
        loss = _finite_number(
            _required(row, "loss", path, row_field),
            path,
            f"{row_field}.loss",
        )
        if loss < 0:
            _schema_error(
                path,
                f"{row_field}.loss",
                "must be non-negative",
            )


def _validate_histories(map_history, map_path, abnn_history, abnn_path):
    map_root = _mapping(map_history, map_path, "root")
    map_rows = _required(map_root, "history", map_path, "")
    _validate_history_rows(map_rows, map_path, "history")
    for index, row_value in enumerate(map_rows):
        row = _mapping(row_value, map_path, f"history[{index}]")
        accuracy = _finite_number(
            _required(
                row,
                "accuracy",
                map_path,
                f"history[{index}]",
            ),
            map_path,
            f"history[{index}].accuracy",
        )
        if not 0.0 <= accuracy <= 1.0:
            _schema_error(
                map_path,
                f"history[{index}].accuracy",
                "expected a value between 0 and 1",
            )

    abnn_root = _mapping(abnn_history, abnn_path, "root")
    members = _sequence(
        _required(abnn_root, "members", abnn_path, ""),
        abnn_path,
        "members",
    )
    if not members:
        _schema_error(abnn_path, "members", "must not be empty")
    for index, member in enumerate(members):
        _validate_history_rows(
            member,
            abnn_path,
            f"members[{index}]",
        )


def _artifact_paths(results_dir: Path, quick: bool):
    suffix = "_quick" if quick else ""
    return (
        results_dir / f"evaluation{suffix}.json",
        results_dir / f"map_history{suffix}.json",
        results_dir / f"abnn_history{suffix}.json",
    )


def _load_artifacts(results_dir, quick: bool):
    results_dir = Path(results_dir)
    evaluation_path, map_path, abnn_path = _artifact_paths(
        results_dir, quick
    )
    evaluation = load_valid_evaluation(evaluation_path)
    map_history = _read_json(map_path)
    abnn_history = _read_json(abnn_path)
    _validate_artifact_mode(evaluation, evaluation_path, quick)
    _validate_artifact_mode(map_history, map_path, quick)
    _validate_artifact_mode(abnn_history, abnn_path, quick)
    _validate_histories(
        map_history,
        map_path,
        abnn_history,
        abnn_path,
    )
    return evaluation, map_history, abnn_history


def _source_digests(results_dir: Path, quick: bool) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in _artifact_paths(results_dir, quick)
    }


def figures_are_current(
    results_dir="results",
    figures_dir="figures",
    quick: bool = False,
) -> bool:
    if not isinstance(quick, bool):
        raise TypeError("quick must be a bool")
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    try:
        _load_artifacts(results_dir, quick)
        sources = _source_digests(results_dir, quick)
        manifest_path = figures_dir / MANIFEST_NAME
        with manifest_path.open(encoding="utf-8") as artifact:
            manifest = json.load(artifact)
        if (
            not isinstance(manifest, dict)
            or manifest.get("mode") != ("quick" if quick else "full")
            or manifest.get("quick") is not quick
            or manifest.get("evaluation_sha256")
            != sources[_artifact_paths(results_dir, quick)[0].name]
            or manifest.get("sources") != sources
        ):
            return False
        outputs = manifest.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != set(OUTPUT_NAMES):
            return False
        for name in OUTPUT_NAMES:
            path = figures_dir / name
            _validate_png(path)
            if outputs[name] != _sha256(path):
                return False
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def _style_axis(axis, *, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(
        axis=grid_axis,
        color="#D5DCE5",
        linewidth=0.8,
        alpha=0.75,
    )
    axis.set_axisbelow(True)


def _plot_architecture():
    figure, axis = plt.subplots(figsize=(12, 6))
    figure.patch.set_facecolor("#F7F9FC")
    axis.set_facecolor("#F7F9FC")
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")

    stages = (
        (0.4, 3.0, 1.6, 1.2, "CIFAR-10\ninput", "#DCEAF7"),
        (2.5, 3.0, 1.8, 1.2, "ResNet-20\nMAP training", "#BCD7EE"),
        (4.8, 3.0, 1.8, 1.2, "Bayesian BN\nconversion", "#FFE0B2"),
        (7.1, 3.0, 1.8, 1.2, "ABNN\nmembers", "#F8C58B"),
        (9.4, 3.0, 1.8, 1.2, "Monte Carlo\nprediction", "#BFE3C0"),
    )
    for x, y, width, height, label, color in stages:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="#334155",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="#172033",
            weight="bold",
        )

    for start, end in ((2.0, 2.5), (4.3, 4.8), (6.6, 7.1), (8.9, 9.4)):
        axis.annotate(
            "",
            xy=(end, 3.6),
            xytext=(start, 3.6),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#526377",
                "linewidth": 1.8,
            },
        )

    output_box = FancyBboxPatch(
        (8.0, 0.65),
        3.2,
        1.15,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.5,
        edgecolor="#334155",
        facecolor="#D7EBD8",
    )
    axis.add_patch(output_box)
    axis.text(
        9.6,
        1.225,
        "Calibration • OOD detection\nPredictive uncertainty",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#172033",
        weight="bold",
    )
    axis.annotate(
        "",
        xy=(9.6, 1.8),
        xytext=(10.3, 3.0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#526377",
            "linewidth": 1.8,
        },
    )

    axis.text(
        8.0,
        6.05,
        "Diverse class-weighted adaptation",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#526377",
    )
    for index, member_y in enumerate((5.65, 5.25, 4.85), start=1):
        axis.plot(
            [7.25, 7.85],
            [member_y, member_y],
            color="#F58518",
            linewidth=5,
            solid_capstyle="round",
            alpha=0.72,
        )
        axis.text(
            8.0,
            member_y,
            f"Member {index}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#526377",
        )
    axis.annotate(
        "",
        xy=(8.0, 4.2),
        xytext=(8.0, 4.68),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#526377",
            "linewidth": 1.4,
        },
    )
    figure.suptitle(
        "Approximate Bayesian Neural Network Pipeline",
        fontsize=18,
        weight="bold",
        color="#172033",
        y=0.96,
    )
    axis.text(
        6,
        6.55,
        "Deterministic representation learning followed by "
        "lightweight Bayesian adaptation",
        ha="center",
        fontsize=10.5,
        color="#526377",
    )
    return figure


def _plot_training_curves(map_history, abnn_history):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    figure.patch.set_facecolor("white")

    map_rows = map_history["history"]
    map_epochs = [row["epoch"] for row in map_rows]
    map_loss = [row["loss"] for row in map_rows]
    axes[0].plot(
        map_epochs,
        map_loss,
        marker="o",
        linewidth=2.2,
        color=METHOD_COLORS["map"],
        label="Training loss",
    )
    axes[0].set_title("MAP optimization", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(frameon=False, loc="best")
    axes[0].xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=True)
    )
    _style_axis(axes[0])

    accuracy_axis = axes[0].twinx()
    accuracy_axis.plot(
        map_epochs,
        [row["accuracy"] for row in map_rows],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color="#7A5195",
        label="Test accuracy",
    )
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0, 1.05)
    accuracy_axis.spines["top"].set_visible(False)
    accuracy_axis.legend(frameon=False, loc="center right")

    for index, rows in enumerate(abnn_history["members"]):
        axes[1].plot(
            [row["epoch"] for row in rows],
            [row["loss"] for row in rows],
            marker="o",
            linewidth=2,
            label=f"Member {index + 1}",
        )
    axes[1].set_title("ABNN member adaptation", weight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weighted cross-entropy loss")
    axes[1].xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=True)
    )
    axes[1].legend(frameon=False, ncols=2)
    _style_axis(axes[1])

    figure.suptitle(
        "Training Dynamics",
        fontsize=16,
        weight="bold",
        color="#172033",
    )
    figure.tight_layout()
    return figure


def _reliability_points(probabilities, targets, bins: int = 15):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=int)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    mean_confidence = []
    empirical_accuracy = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            mask |= confidence == 0
        if mask.any():
            mean_confidence.append(float(confidence[mask].mean()))
            empirical_accuracy.append(
                float((predictions[mask] == targets[mask]).mean())
            )
    return np.asarray(mean_confidence), np.asarray(empirical_accuracy)


def _plot_reliability(evaluation):
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color="#6B7280",
        label="Perfect calibration",
    )
    for method_name in METHODS:
        method = evaluation["methods"][method_name]
        confidence, accuracy = _reliability_points(
            method["probabilities"],
            method["targets"],
            bins=15,
        )
        axis.plot(
            confidence,
            accuracy,
            marker="o",
            markersize=5,
            linewidth=2,
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted confidence",
        ylabel="Empirical accuracy",
        title="Reliability Diagram (15 bins)",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis, grid_axis="both")
    figure.tight_layout()
    return figure


def _shared_entropy_bins(evaluation, method_name):
    method = evaluation["methods"][method_name]
    arrays = [
        np.asarray(method["predictive_entropy"], dtype=float),
        *(
            np.asarray(
                method["ood_predictive_entropy"][dataset],
                dtype=float,
            )
            for dataset in OOD_DATASETS
        ),
    ]
    upper = max(float(array.max()) for array in arrays)
    if upper <= 0:
        upper = 1.0
    return np.linspace(0.0, upper * 1.03, 21)


def _plot_ood_entropy(evaluation):
    figure, axes = plt.subplots(
        1,
        len(METHODS),
        figsize=(14.4, 4.6),
        sharey=True,
    )
    dataset_styles = (
        ("predictive_entropy", "CIFAR-10 (ID)", "#4C78A8"),
        ("svhn", "SVHN (OOD)", "#E45756"),
        ("cifar100", "CIFAR-100 (OOD)", "#72B7B2"),
    )
    for axis, method_name in zip(axes, METHODS):
        method = evaluation["methods"][method_name]
        bins = _shared_entropy_bins(evaluation, method_name)
        for dataset, label, color in dataset_styles:
            values = (
                method["predictive_entropy"]
                if dataset == "predictive_entropy"
                else method["ood_predictive_entropy"][dataset]
            )
            axis.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.48,
                color=color,
                label=label,
                edgecolor="white",
                linewidth=0.4,
            )
        axis.set_title(METHOD_LABELS[method_name], weight="bold")
        axis.set_xlabel("Predictive entropy")
        _style_axis(axis)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Density")
    figure.suptitle(
        "ID and OOD Predictive Entropy",
        fontsize=16,
        weight="bold",
        color="#172033",
    )
    figure.tight_layout()
    return figure


def _annotate_bars(axis, containers) -> None:
    for container in containers:
        axis.bar_label(
            container,
            fmt="%.2f",
            padding=2,
            fontsize=7.5,
            rotation=90,
        )


def _grouped_bars(axis, categories, values, title):
    x_positions = np.arange(len(categories))
    width = 0.24
    containers = []
    for index, method_name in enumerate(METHODS):
        offset = (index - 1) * width
        container = axis.bar(
            x_positions + offset,
            values[method_name],
            width,
            color=METHOD_COLORS[method_name],
            label=METHOD_LABELS[method_name],
        )
        containers.append(container)
    axis.set_xticks(x_positions, categories)
    axis.set_ylim(0, 1.12)
    axis.set_title(title, weight="bold")
    axis.set_ylabel("Score")
    _style_axis(axis)
    _annotate_bars(axis, containers)


def _plot_metrics_comparison(evaluation):
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    classification_values = {
        method_name: [
            evaluation["methods"][method_name]["classification"][metric]
            for metric in ("accuracy", "ece")
        ]
        for method_name in METHODS
    }
    _grouped_bars(
        axes[0],
        ("Accuracy ↑", "ECE ↓"),
        classification_values,
        "CIFAR-10 classification and calibration",
    )

    ood_categories = []
    for dataset in OOD_DATASETS:
        ood_categories.extend(
            (f"{OOD_LABELS[dataset]}\nAUROC ↑", f"{OOD_LABELS[dataset]}\nFPR95 ↓")
        )
    ood_values = {}
    for method_name in METHODS:
        values = []
        for dataset in OOD_DATASETS:
            metrics = evaluation["methods"][method_name]["ood"][dataset]
            values.extend((metrics["auroc"], metrics["fpr95"]))
        ood_values[method_name] = values
    _grouped_bars(
        axes[1],
        ood_categories,
        ood_values,
        "OOD detection",
    )
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(-0.1, -0.17),
        ncols=3,
    )
    figure.suptitle(
        "Predictive Performance Comparison",
        fontsize=16,
        weight="bold",
        color="#172033",
    )
    figure.tight_layout()
    return figure


def _uncertainty_records(evaluation):
    images = np.asarray(evaluation["example_images"], dtype=float)
    targets = np.asarray(evaluation["example_targets"], dtype=int)
    method = evaluation["methods"]["abnn_full"]
    count = len(images)
    probabilities = np.asarray(method["probabilities"], dtype=float)[:count]
    entropies = np.asarray(method["predictive_entropy"], dtype=float)[:count]
    predictions = probabilities.argmax(axis=1)
    order = np.argsort(entropies, kind="stable")
    return [
        {
            "image": images[index],
            "target": int(targets[index]),
            "prediction": int(predictions[index]),
            "entropy": float(entropies[index]),
        }
        for index in order
    ]


def _class_label(index: int, class_count: int) -> str:
    if class_count == len(CIFAR10_CLASSES):
        return CIFAR10_CLASSES[index]
    return str(index)


def _plot_uncertainty_examples(evaluation):
    records = _uncertainty_records(evaluation)
    columns = min(4, len(records))
    rows = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns, 3.2 * rows),
        squeeze=False,
    )
    class_count = len(
        evaluation["methods"]["abnn_full"]["probabilities"][0]
    )
    for axis, record in zip(axes.flat, records):
        image = np.moveaxis(record["image"], 0, -1)
        if image.shape[-1] == 1:
            image = image[..., 0]
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        else:
            axis.imshow(image)
        true_label = _class_label(record["target"], class_count)
        predicted_label = _class_label(
            record["prediction"],
            class_count,
        )
        correct = record["target"] == record["prediction"]
        axis.set_title(
            f"True: {true_label}  |  Pred: {predicted_label}\n"
            f"Entropy: {record['entropy']:.3f}",
            fontsize=9,
            color="#256D3D" if correct else "#B33A3A",
            weight="bold",
        )
        axis.axis("off")
    for axis in axes.flat[len(records) :]:
        axis.axis("off")
    figure.suptitle(
        "ABNN Ensemble Uncertainty Examples\n"
        "Lowest → highest predictive entropy",
        fontsize=15,
        weight="bold",
        color="#172033",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def make_all_figures(
    results_dir="results",
    figures_dir="figures",
    quick: bool = False,
) -> list[Path]:
    """Build the six report figures from quick or full JSON artifacts."""
    if not isinstance(quick, bool):
        raise TypeError("quick must be a bool")
    results_dir = Path(results_dir)
    source_digests = _source_digests(results_dir, quick)
    evaluation, map_history, abnn_history = _load_artifacts(
        results_dir,
        quick,
    )
    if _source_digests(results_dir, quick) != source_digests:
        raise RuntimeError("figure source artifacts changed while loading")
    figures_dir = Path(figures_dir)
    builders = (
        _plot_architecture,
        lambda: _plot_training_curves(map_history, abnn_history),
        lambda: _plot_reliability(evaluation),
        lambda: _plot_ood_entropy(evaluation),
        lambda: _plot_metrics_comparison(evaluation),
        lambda: _plot_uncertainty_examples(evaluation),
    )
    figures_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{figures_dir.name}-",
        dir=figures_dir.parent,
    ) as temporary:
        staging_dir = Path(temporary)
        staged_paths = []
        for builder, name in zip(builders, OUTPUT_NAMES):
            staged_paths.append(_save(builder(), staging_dir / name))
        for path in staged_paths:
            _validate_png(path)
        output_digests = {
            path.name: _sha256(path) for path in staged_paths
        }
        if _source_digests(results_dir, quick) != source_digests:
            raise RuntimeError(
                "figure source artifacts changed during plotting"
            )
        figures_dir.mkdir(parents=True, exist_ok=True)
        for path in staged_paths:
            path.replace(figures_dir / path.name)

    if _source_digests(results_dir, quick) != source_digests:
        raise RuntimeError("figure source artifacts changed before publishing")
    evaluation_path = _artifact_paths(results_dir, quick)[0]
    save_json(
        figures_dir / MANIFEST_NAME,
        {
            "mode": "quick" if quick else "full",
            "quick": quick,
            "evaluation_sha256": source_digests[evaluation_path.name],
            "sources": source_digests,
            "outputs": output_digests,
        },
    )
    return [figures_dir / name for name in OUTPUT_NAMES]

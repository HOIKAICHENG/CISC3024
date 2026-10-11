# ABNN Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, document, and publish a reproducible CPU-scale ABNN image-classification and uncertainty experiment for CISC3024 AI Assignment #2.

**Architecture:** A CIFAR ResNet-20 is first trained as a deterministic MAP classifier, then copied into three models whose BatchNorm layers are replaced by Bayesian Normalization Layers. Only normalization affine parameters are adapted; evaluation averages stochastic predictions and compares MAP, one-member ABNN, and full ABNN on calibration and OOD detection.

**Tech Stack:** Python 3.12, PyTorch 2.13 CPU, torchvision 0.28, NumPy, scikit-learn, Matplotlib, Pillow, pytest, python-docx, and Microsoft Word COM conversion on Windows.

## Global Constraints

- Work only under `AI Assignment2/` except for repository-level Git commands.
- Student identity is `HOI KAI CHENG`, student ID `UC325381`.
- Use CIFAR-10 train/test as in-distribution data and SVHN test plus CIFAR-100 test as OOD data.
- Full MAP run: seed 0, ResNet-20, 20 epochs, batch 128, SGD learning rate 0.1, momentum 0.9, weight decay 0.0005, one warm-up epoch, then cosine decay.
- Full ABNN run: member seeds 101/102/103, five epochs per member, SGD learning rate 0.01, momentum 0.9, no weight decay, three emphasized classes per member at class weight 5.
- Full inference uses three members and three stochastic samples per member.
- Quick mode uses 1,000 CIFAR-10 training examples, 500 examples per evaluation dataset, one MAP epoch, one ABNN member, one adaptation epoch, and two stochastic samples.
- Every reported number must be read from persisted JSON output; paper values and assignment values must be labeled separately.
- Do not claim a full reproduction of the paper's ResNet-50/WideResNet results.
- Final artifacts are DOCX and PDF, and the repository target is `https://github.com/HOIKAICHENG/CISC3024.git`.

---

## File Map

- `requirements.txt`: reproducible dependencies.
- `.gitignore`: downloaded data, model checkpoints, caches, Office temporary files.
- `code/config.py`: approved settings and deterministic seeding.
- `code/data.py`: CIFAR-10, CIFAR-100, and SVHN loaders.
- `code/model.py`: CIFAR ResNet-20.
- `code/abnn.py`: Bayesian normalization, conversion, freezing, class diversity, and stochastic prediction.
- `code/metrics.py`: classification, calibration, and OOD metrics.
- `code/engine.py`: reusable train/evaluate/checkpoint functions.
- `code/train_map.py`: deterministic training entry point.
- `code/train_abnn.py`: ABNN adaptation entry point.
- `code/evaluate.py`: method comparison and JSON output.
- `code/make_figures.py`: figures generated only from saved artifacts.
- `code/run_all.py`: resumable quick/full orchestration.
- `tests/`: focused unit and integration tests.
- `build_report.py`: DOCX/PDF report generation.
- `README.md`: reproduction instructions and source-code webpage.
- `results/`, `figures/`, `report/`: generated experiment and submission artifacts.

### Task 1: Configuration, dependencies, and deterministic data loading

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `code/__init__.py`
- Create: `code/config.py`
- Create: `code/data.py`
- Test: `tests/test_config_data.py`

**Interfaces:**
- Produces: `ExperimentConfig`, `seed_everything(seed: int) -> None`, `build_loaders(config: ExperimentConfig, quick: bool) -> dict[str, DataLoader]`.
- The loader dictionary keys are `train`, `cifar10_test`, `svhn_test`, and `cifar100_test`.

- [ ] **Step 1: Write failing configuration and subset tests**

```python
from code.config import ExperimentConfig
from code.data import deterministic_subset_indices


def test_approved_full_defaults():
    cfg = ExperimentConfig()
    assert (cfg.map_epochs, cfg.abnn_epochs, cfg.batch_size) == (20, 5, 128)
    assert (cfg.member_seeds, cfg.mc_samples) == ((101, 102, 103), 3)


def test_subset_indices_are_seeded_and_bounded():
    a = deterministic_subset_indices(100, 12, seed=7)
    b = deterministic_subset_indices(100, 12, seed=7)
    assert a == b
    assert len(a) == len(set(a)) == 12
    assert min(a) >= 0 and max(a) < 100
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `python -m pytest tests/test_config_data.py -v`

Expected: FAIL because `code.config` and `code.data` do not exist.

- [ ] **Step 3: Implement the approved configuration and deterministic loaders**

```python
# code/config.py
from dataclasses import asdict, dataclass
import random
import numpy as np
import torch


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    member_seeds: tuple[int, ...] = (101, 102, 103)
    batch_size: int = 128
    map_epochs: int = 20
    abnn_epochs: int = 5
    map_lr: float = 0.1
    abnn_lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    mc_samples: int = 3
    emphasized_classes: int = 3
    emphasis_factor: float = 5.0
    num_workers: int = 2
    data_dir: str = "data"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
```

```python
# code/data.py
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .config import ExperimentConfig

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def deterministic_subset_indices(size: int, count: int, seed: int) -> list[int]:
    if not 0 < count <= size:
        raise ValueError(f"subset count {count} must be in [1, {size}]")
    return random.Random(seed).sample(range(size), count)


def build_loaders(config: ExperimentConfig, quick: bool = False) -> dict[str, DataLoader]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    root = config.data_dir
    sets = {
        "train": datasets.CIFAR10(root, train=True, download=True, transform=train_tf),
        "cifar10_test": datasets.CIFAR10(root, train=False, download=True, transform=test_tf),
        "svhn_test": datasets.SVHN(root, split="test", download=True, transform=test_tf),
        "cifar100_test": datasets.CIFAR100(root, train=False, download=True, transform=test_tf),
    }
    if quick:
        limits = {"train": 1000, "cifar10_test": 500, "svhn_test": 500, "cifar100_test": 500}
        sets = {
            name: Subset(ds, deterministic_subset_indices(len(ds), limits[name], config.seed))
            for name, ds in sets.items()
        }
    return {
        name: DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=name == "train",
            num_workers=config.num_workers,
            pin_memory=False,
        )
        for name, ds in sets.items()
    }
```

- [ ] **Step 4: Add dependencies and ignore generated heavy files**

```text
# requirements.txt
torch>=2.13.0
torchvision>=0.28.0
numpy>=2.0
scipy>=1.14
scikit-learn>=1.5
matplotlib>=3.9
Pillow>=10.0
python-docx>=1.1
pywin32>=306
pytest>=8.0
```

```gitignore
data/
results/*.pt
results/*.pth
__pycache__/
*.pyc
.pytest_cache/
~$*
*.tmp
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_config_data.py -v`

Expected: 2 passed.

Commit:

```bash
git add "AI Assignment2/requirements.txt" "AI Assignment2/.gitignore" "AI Assignment2/code" "AI Assignment2/tests/test_config_data.py"
git commit -m "feat: add ABNN experiment configuration and data loaders"
```

### Task 2: CIFAR ResNet-20 backbone

**Files:**
- Create: `code/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `resnet20(num_classes: int = 10) -> CifarResNet`.
- `CifarResNet.forward(x)` returns logits shaped `[batch, num_classes]`.

- [ ] **Step 1: Write failing shape and model-size tests**

```python
import torch
from code.model import resnet20


def test_resnet20_output_and_size():
    model = resnet20()
    output = model(torch.randn(4, 3, 32, 32))
    assert output.shape == (4, 10)
    assert 250_000 < sum(p.numel() for p in model.parameters()) < 300_000
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_model.py -v`

Expected: FAIL because `code.model` does not exist.

- [ ] **Step 3: Implement the CIFAR residual network**

```python
# code/model.py
import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, blocks=(3, 3, 3), num_classes: int = 10):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._layer(16, blocks[0], 1)
        self.layer2 = self._layer(32, blocks[1], 2)
        self.layer3 = self._layer(64, blocks[2], 2)
        self.fc = nn.Linear(64, num_classes)
        self.apply(self._init)

    def _layer(self, channels: int, count: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        layers = []
        for current_stride in strides:
            layers.append(BasicBlock(self.in_channels, channels, current_stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def resnet20(num_classes: int = 10) -> CifarResNet:
    return CifarResNet((3, 3, 3), num_classes)
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_model.py -v`

Expected: 1 passed.

Commit:

```bash
git add "AI Assignment2/code/model.py" "AI Assignment2/tests/test_model.py"
git commit -m "feat: implement CIFAR ResNet-20 backbone"
```

### Task 3: Bayesian Normalization Layer and ABNN conversion

**Files:**
- Create: `code/abnn.py`
- Test: `tests/test_abnn.py`

**Interfaces:**
- Produces: `BayesianBatchNorm2d`, `convert_to_abnn(model)`, `freeze_except_bayesian_norm(model)`, `set_stochastic(model, enabled)`, and `class_weights_for_member(...)`.
- Conversion mutates and returns the supplied model.

- [ ] **Step 1: Write failing conversion, freezing, and stochasticity tests**

```python
import copy
import torch
from torch import nn
from code.abnn import (
    BayesianBatchNorm2d,
    convert_to_abnn,
    freeze_except_bayesian_norm,
    set_stochastic,
)


def test_conversion_preserves_deterministic_prediction_and_freezes_backbone():
    source = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Flatten())
    source.eval()
    x = torch.randn(2, 3, 5, 5)
    expected = source(x)
    converted = convert_to_abnn(copy.deepcopy(source)).eval()
    set_stochastic(converted, False)
    assert torch.allclose(converted(x), expected, atol=1e-6)
    freeze_except_bayesian_norm(converted)
    trainable = [name for name, value in converted.named_parameters() if value.requires_grad]
    assert trainable == ["1.weight", "1.bias"]


def test_enabled_bayesian_layer_changes_predictions():
    layer = BayesianBatchNorm2d(3).eval()
    x = torch.ones(2, 3, 4, 4)
    layer.stochastic = True
    assert not torch.equal(layer(x), layer(x))
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_abnn.py -v`

Expected: FAIL because `code.abnn` does not exist.

- [ ] **Step 3: Implement BNL and recursive conversion**

```python
# code/abnn.py
import random
import torch
from torch import nn
from torch.nn import functional as F


class BayesianBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, **kwargs):
        super().__init__(num_features, **kwargs)
        self.stochastic = True

    @classmethod
    def from_batchnorm(cls, source: nn.BatchNorm2d) -> "BayesianBatchNorm2d":
        target = cls(
            source.num_features,
            eps=source.eps,
            momentum=source.momentum,
            affine=source.affine,
            track_running_stats=source.track_running_stats,
        )
        target.load_state_dict(source.state_dict())
        return target

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
        normalized = F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training or not self.track_running_stats,
            self.momentum if self.momentum is not None else 0.0,
            self.eps,
        )
        scale = self.weight.view(1, -1, 1, 1)
        if self.stochastic:
            scale = scale * (1.0 + torch.randn_like(scale))
        return normalized * scale + self.bias.view(1, -1, 1, 1)


def convert_to_abnn(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, BayesianBatchNorm2d):
            setattr(module, name, BayesianBatchNorm2d.from_batchnorm(child))
        else:
            convert_to_abnn(child)
    return module


def freeze_except_bayesian_norm(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, BayesianBatchNorm2d):
            module.weight.requires_grad = True
            module.bias.requires_grad = True


def set_stochastic(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, BayesianBatchNorm2d):
            module.stochastic = enabled


def class_weights_for_member(
    num_classes: int,
    emphasized: int,
    factor: float,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    selected = sorted(random.Random(seed).sample(range(num_classes), emphasized))
    weights = torch.ones(num_classes)
    weights[selected] = factor
    return weights, selected
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_abnn.py -v`

Expected: 2 passed.

Commit:

```bash
git add "AI Assignment2/code/abnn.py" "AI Assignment2/tests/test_abnn.py"
git commit -m "feat: add Bayesian normalization and ABNN conversion"
```

### Task 4: Calibration and OOD metrics

**Files:**
- Create: `code/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `classification_metrics(probs, targets, bins=15) -> dict[str, float]`, `predictive_entropy(probs)`, and `ood_metrics(id_scores, ood_scores)`.
- Higher OOD score must mean "more likely OOD".

- [ ] **Step 1: Write hand-checkable failing tests**

```python
import numpy as np
from code.metrics import classification_metrics, ood_metrics, predictive_entropy


def test_perfect_binary_predictions():
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = classification_metrics(probs, np.array([0, 1]), bins=2)
    assert result["accuracy"] == 1.0
    assert np.isclose(result["nll"], -np.log(0.9))
    assert np.isclose(result["brier"], 0.02)


def test_separated_ood_scores_are_perfect():
    result = ood_metrics(np.array([0.1, 0.2]), np.array([0.8, 0.9]))
    assert result["auroc"] == 1.0
    assert result["aupr"] == 1.0
    assert result["fpr95"] == 0.0


def test_uniform_prediction_has_more_entropy():
    assert predictive_entropy(np.array([[0.5, 0.5]]))[0] > predictive_entropy(
        np.array([[0.99, 0.01]])
    )[0]
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: FAIL because `code.metrics` does not exist.

- [ ] **Step 3: Implement numerically stable metrics**

```python
# code/metrics.py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.clip(probs, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def classification_metrics(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> dict[str, float]:
    safe = np.clip(probs, 1e-12, 1.0)
    predictions = safe.argmax(axis=1)
    confidence = safe.max(axis=1)
    accuracy = (predictions == targets).mean()
    nll = -np.log(safe[np.arange(len(targets)), targets]).mean()
    one_hot = np.eye(safe.shape[1])[targets]
    brier = np.square(safe - one_hot).sum(axis=1).mean()
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs((predictions[mask] == targets[mask]).mean() - confidence[mask].mean())
    return {"accuracy": float(accuracy), "nll": float(nll), "brier": float(brier), "ece": float(ece)}


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    index = np.flatnonzero(tpr >= 0.95)
    fpr95 = fpr[index[0]] if len(index) else 1.0
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "fpr95": float(fpr95),
    }
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_metrics.py -v`

Expected: 3 passed.

Commit:

```bash
git add "AI Assignment2/code/metrics.py" "AI Assignment2/tests/test_metrics.py"
git commit -m "feat: add calibration and OOD metrics"
```

### Task 5: Shared training engine and MAP training

**Files:**
- Create: `code/engine.py`
- Create: `code/train_map.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Produces: `train_epoch`, `deterministic_probabilities`, `save_training_state`, `load_training_state`, and `run_map_training(config, loaders, quick=False)`.
- MAP artifacts: `results/map_checkpoint.pt` and `results/map_history.json`; quick artifacts use a `_quick` suffix.

- [ ] **Step 1: Write a failing synthetic train/checkpoint test**

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from code.engine import load_training_state, save_training_state, train_epoch


def test_train_epoch_and_checkpoint_round_trip(tmp_path):
    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.randn(8, 2), torch.tensor([0, 1] * 4)), batch_size=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert loss > 0
    path = tmp_path / "state.pt"
    save_training_state(path, model, optimizer, epoch=1, history=[{"epoch": 1, "loss": loss}])
    restored = nn.Linear(2, 2)
    state = load_training_state(path, restored)
    assert state["epoch"] == 1
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_engine.py -v`

Expected: FAIL because `code.engine` does not exist.

- [ ] **Step 3: Implement shared epoch, prediction, JSON, and checkpoint helpers**

```python
# code/engine.py
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
    return total_loss / total


@torch.inference_mode()
def deterministic_probabilities(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    for inputs, labels in loader:
        probabilities.append(model(inputs.to(device)).softmax(1).cpu())
        targets.append(labels)
    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def save_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_training_state(path, model, optimizer, epoch: int, history: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "history": history}, path)


def load_training_state(path, model, optimizer=None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
```

- [ ] **Step 4: Implement resumable MAP training**

```python
# code/train_map.py
from pathlib import Path
import time
import math
import torch
from torch import nn
from .config import ExperimentConfig, seed_everything
from .engine import deterministic_probabilities, load_training_state, save_json, save_training_state, train_epoch
from .metrics import classification_metrics
from .model import resnet20


def run_map_training(config: ExperimentConfig, loaders, quick: bool = False):
    seed_everything(config.seed)
    device = torch.device("cpu")
    model = resnet20().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.map_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    epochs = 1 if quick else config.map_epochs
    suffix = "_quick" if quick else ""
    checkpoint = Path(config.results_dir) / f"map_checkpoint{suffix}.pt"
    history_path = Path(config.results_dir) / f"map_history{suffix}.json"
    history, start = [], 0
    if checkpoint.exists():
        state = load_training_state(checkpoint, model, optimizer)
        history, start = state["history"], state["epoch"]
    def lr_factor(epoch: int) -> float:
        if epoch == 0:
            return 0.1
        progress = (epoch - 1) / max(1, epochs - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    for _ in range(start):
        scheduler.step()
    started = time.perf_counter()
    for epoch in range(start, epochs):
        loss = train_epoch(model, loaders["train"], optimizer, nn.CrossEntropyLoss(), device)
        probs, targets = deterministic_probabilities(model, loaders["cifar10_test"], device)
        row = {"epoch": epoch + 1, "loss": loss, "lr": optimizer.param_groups[0]["lr"], **classification_metrics(probs, targets)}
        history.append(row)
        scheduler.step()
        save_training_state(checkpoint, model, optimizer, epoch + 1, history)
        save_json(history_path, {"config": config.to_dict(), "history": history, "elapsed_seconds": time.perf_counter() - started})
    return model, history
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_engine.py -v`

Expected: 1 passed.

Commit:

```bash
git add "AI Assignment2/code/engine.py" "AI Assignment2/code/train_map.py" "AI Assignment2/tests/test_engine.py"
git commit -m "feat: add resumable MAP training"
```

### Task 6: ABNN member adaptation and Monte Carlo prediction

**Files:**
- Modify: `code/abnn.py`
- Create: `code/train_abnn.py`
- Test: `tests/test_abnn_training.py`

**Interfaces:**
- Produces: `monte_carlo_probabilities(models, loader, samples, device)` and `run_abnn_training(config, loaders, map_checkpoint, quick=False)`.
- Member artifacts: `results/abnn_member_{index}.pt` and `results/abnn_history.json`, with `_quick` suffixes.

- [ ] **Step 1: Write a failing probability-aggregation test**

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from code.abnn import monte_carlo_probabilities


def test_monte_carlo_probabilities_are_normalized():
    models = [nn.Linear(2, 3), nn.Linear(2, 3)]
    loader = DataLoader(TensorDataset(torch.randn(5, 2), torch.arange(5) % 3), batch_size=2)
    probs, targets, sample_probs = monte_carlo_probabilities(models, loader, samples=2, device=torch.device("cpu"))
    assert probs.shape == (5, 3)
    assert sample_probs.shape == (4, 5, 3)
    assert targets.shape == (5,)
    assert torch.allclose(torch.from_numpy(probs.sum(1)), torch.ones(5), atol=1e-6)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_abnn_training.py -v`

Expected: FAIL because `monte_carlo_probabilities` is missing.

- [ ] **Step 3: Implement stochastic aggregation**

```python
# append to code/abnn.py
import numpy as np


@torch.inference_mode()
def monte_carlo_probabilities(models, loader, samples: int, device):
    draws, saved_targets = [], None
    for model in models:
        model.eval()
        set_stochastic(model, True)
        for _ in range(samples):
            batches, targets = [], []
            for inputs, labels in loader:
                batches.append(model(inputs.to(device)).softmax(1).cpu())
                targets.append(labels)
            draws.append(torch.cat(batches))
            current_targets = torch.cat(targets)
            if saved_targets is None:
                saved_targets = current_targets
            elif not torch.equal(saved_targets, current_targets):
                raise RuntimeError("evaluation loader order changed between Monte Carlo passes")
    sample_probs = torch.stack(draws).numpy()
    return sample_probs.mean(axis=0), saved_targets.numpy(), sample_probs
```

- [ ] **Step 4: Implement seeded ABNN member adaptation**

```python
# code/train_abnn.py
import copy
from pathlib import Path
import torch
from torch import nn
from .abnn import class_weights_for_member, convert_to_abnn, freeze_except_bayesian_norm
from .config import ExperimentConfig, seed_everything
from .engine import load_training_state, save_json, save_training_state, train_epoch
from .model import resnet20


def run_abnn_training(config: ExperimentConfig, loaders, map_checkpoint, quick: bool = False):
    base = resnet20()
    load_training_state(map_checkpoint, base)
    seeds = config.member_seeds[:1] if quick else config.member_seeds
    epochs = 1 if quick else config.abnn_epochs
    suffix = "_quick" if quick else ""
    members, histories = [], []
    for index, seed in enumerate(seeds):
        seed_everything(seed)
        model = convert_to_abnn(copy.deepcopy(base))
        freeze_except_bayesian_norm(model)
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.abnn_lr,
            momentum=config.momentum,
        )
        weights, selected = class_weights_for_member(10, config.emphasized_classes, config.emphasis_factor, seed)
        criterion = nn.CrossEntropyLoss(weight=weights)
        checkpoint = Path(config.results_dir) / f"abnn_member_{index}{suffix}.pt"
        history, start = [], 0
        if checkpoint.exists():
            state = load_training_state(checkpoint, model, optimizer)
            history, start = state["history"], state["epoch"]
        for epoch in range(start, epochs):
            loss = train_epoch(model, loaders["train"], optimizer, criterion, torch.device("cpu"))
            history.append({"epoch": epoch + 1, "loss": loss, "emphasized_classes": selected})
            save_training_state(checkpoint, model, optimizer, epoch + 1, history)
        members.append(model)
        histories.append(history)
    save_json(Path(config.results_dir) / f"abnn_history{suffix}.json", {"config": config.to_dict(), "members": histories})
    return members, histories
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_abnn.py tests/test_abnn_training.py -v`

Expected: all tests pass.

Commit:

```bash
git add "AI Assignment2/code/abnn.py" "AI Assignment2/code/train_abnn.py" "AI Assignment2/tests/test_abnn_training.py"
git commit -m "feat: add ABNN adaptation and stochastic inference"
```

### Task 7: Comparative evaluation and persisted results

**Files:**
- Create: `code/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `evaluate_methods(config, loaders, map_model, members, quick=False) -> dict`.
- Writes `results/evaluation.json` or `results/evaluation_quick.json`.
- Each method contains classification metrics, per-example probabilities/targets, and OOD metrics for both datasets.

- [ ] **Step 1: Write a failing evaluation-schema test using tiny models**

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from code.config import ExperimentConfig
from code.evaluate import evaluate_methods


def test_evaluation_contains_all_methods_and_ood_sets(tmp_path):
    loader = DataLoader(TensorDataset(torch.randn(6, 3, 4, 4), torch.arange(6) % 2), batch_size=3)
    cfg = ExperimentConfig(results_dir=str(tmp_path), mc_samples=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    result = evaluate_methods(
        cfg,
        {"cifar10_test": loader, "svhn_test": loader, "cifar100_test": loader},
        model,
        [nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))],
        quick=True,
    )
    assert set(result["methods"]) == {"map", "abnn_single", "abnn_full"}
    assert set(result["methods"]["abnn_full"]["ood"]) == {"svhn", "cifar100"}
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_evaluate.py -v`

Expected: FAIL because `code.evaluate` does not exist.

- [ ] **Step 3: Implement evaluation with JSON-safe arrays**

```python
# code/evaluate.py
from pathlib import Path
import numpy as np
import torch
from .abnn import monte_carlo_probabilities
from .engine import deterministic_probabilities, save_json
from .metrics import classification_metrics, ood_metrics, predictive_entropy


def _method_result(id_probs, targets, ood_probabilities):
    id_entropy = predictive_entropy(id_probs)
    return {
        "classification": classification_metrics(id_probs, targets),
        "ood": {
            name: ood_metrics(id_entropy, predictive_entropy(probs))
            for name, probs in ood_probabilities.items()
        },
        "ood_predictive_entropy": {
            name: predictive_entropy(probs).tolist()
            for name, probs in ood_probabilities.items()
        },
        "probabilities": id_probs.tolist(),
        "targets": targets.tolist(),
        "predictive_entropy": id_entropy.tolist(),
    }


def _example_images(loader, count: int = 16):
    images, targets = next(iter(loader))
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    images = (images[:count] * std + mean).clamp(0, 1)
    return images.tolist(), targets[:count].tolist()


def evaluate_methods(config, loaders, map_model, members, quick: bool = False):
    device = torch.device("cpu")
    samples = 2 if quick else config.mc_samples
    map_id, targets = deterministic_probabilities(map_model, loaders["cifar10_test"], device)
    map_ood = {
        "svhn": deterministic_probabilities(map_model, loaders["svhn_test"], device)[0],
        "cifar100": deterministic_probabilities(map_model, loaders["cifar100_test"], device)[0],
    }
    methods = {"map": _method_result(map_id, targets, map_ood)}
    for name, selected in (("abnn_single", members[:1]), ("abnn_full", members)):
        id_probs, member_targets, _ = monte_carlo_probabilities(selected, loaders["cifar10_test"], samples, device)
        if not np.array_equal(targets, member_targets):
            raise RuntimeError("MAP and ABNN target order differs")
        ood = {
            key: monte_carlo_probabilities(selected, loaders[f"{key}_test"], samples, device)[0]
            for key in ("svhn", "cifar100")
        }
        methods[name] = _method_result(id_probs, targets, ood)
    images, image_targets = _example_images(loaders["cifar10_test"])
    payload = {
        "config": config.to_dict(),
        "methods": methods,
        "example_images": images,
        "example_targets": image_targets,
    }
    suffix = "_quick" if quick else ""
    save_json(Path(config.results_dir) / f"evaluation{suffix}.json", payload)
    return payload
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_evaluate.py -v`

Expected: 1 passed.

Commit:

```bash
git add "AI Assignment2/code/evaluate.py" "AI Assignment2/tests/test_evaluate.py"
git commit -m "feat: evaluate ABNN calibration and OOD detection"
```

### Task 8: Reproducible figures

**Files:**
- Create: `code/make_figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Produces: `make_all_figures(results_dir="results", figures_dir="figures", quick=False) -> list[Path]`.
- Required outputs: `architecture.png`, `training_curves.png`, `reliability.png`, `ood_entropy.png`, `metrics_comparison.png`, and `uncertainty_examples.png`.

- [ ] **Step 1: Write a failing artifact-generation test**

```python
import json
from pathlib import Path
from code.make_figures import make_all_figures


def test_figures_are_nonempty(tmp_path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    fixture = {
        "methods": {
            name: {
                "classification": {"accuracy": 0.5, "nll": 1.0, "brier": 0.8, "ece": 0.1},
                "ood": {"svhn": {"auroc": 0.7, "aupr": 0.7, "fpr95": 0.8}, "cifar100": {"auroc": 0.6, "aupr": 0.6, "fpr95": 0.9}},
                "ood_predictive_entropy": {"svhn": [0.8, 0.9], "cifar100": [0.7, 0.8]},
                "probabilities": [[0.8, 0.2], [0.4, 0.6]],
                "targets": [0, 1],
                "predictive_entropy": [0.5, 0.67],
            }
            for name in ("map", "abnn_single", "abnn_full")
        },
        "example_images": [[[[0.0] * 32 for _ in range(32)]] * 3, [[[1.0] * 32 for _ in range(32)]] * 3],
        "example_targets": [0, 1],
    }
    (results / "evaluation_quick.json").write_text(json.dumps(fixture), encoding="utf-8")
    (results / "map_history_quick.json").write_text(json.dumps({"history": [{"epoch": 1, "loss": 1.0, "accuracy": 0.5}]}), encoding="utf-8")
    (results / "abnn_history_quick.json").write_text(json.dumps({"members": [[{"epoch": 1, "loss": 0.9}]]}), encoding="utf-8")
    paths = make_all_figures(results, figures, quick=True)
    assert len(paths) == 6
    assert all(path.stat().st_size > 1000 for path in paths)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_figures.py -v`

Expected: FAIL because `code.make_figures` does not exist.

- [ ] **Step 3: Implement six headless Matplotlib figures**

Implement `make_all_figures` with the non-interactive `Agg` backend. It must:

```python
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
```

Use explicit plotting functions for:

- a boxes-and-arrows architecture schematic;
- MAP and member loss curves;
- 15-bin confidence versus empirical accuracy reliability curves;
- ID/OOD entropy histograms using evaluation arrays saved by Task 7;
- grouped bars for accuracy/ECE and AUROC/FPR95;
- a labeled uncertainty grid using `example_images`, `example_targets`, and the
  matching first entries of `abnn_full.predictive_entropy`, sorted from lowest
  to highest entropy and annotated with true/predicted class and entropy.

No figure may contain hard-coded experimental scores.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_figures.py -v`

Expected: 1 passed and six PNGs created in the temporary directory.

Commit:

```bash
git add "AI Assignment2/code/make_figures.py" "AI Assignment2/code/evaluate.py" "AI Assignment2/tests/test_figures.py"
git commit -m "feat: generate ABNN experiment figures"
```

### Task 9: End-to-end orchestration and quick smoke workflow

**Files:**
- Create: `code/run_all.py`
- Test: `tests/test_run_all.py`

**Interfaces:**
- Produces CLI `python -m code.run_all [--quick] [--skip-report]`.
- Existing complete checkpoints are resumed; existing complete stage outputs are reused.

- [ ] **Step 1: Write a failing CLI help test**

```python
import subprocess
import sys


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
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_run_all.py -v`

Expected: FAIL because `code.run_all` does not exist.

- [ ] **Step 3: Implement orchestration**

```python
# code/run_all.py
import argparse
from pathlib import Path
from .config import ExperimentConfig
from .data import build_loaders
from .evaluate import evaluate_methods
from .make_figures import make_all_figures
from .train_abnn import run_abnn_training
from .train_map import run_map_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ABNN assignment experiment")
    parser.add_argument("--quick", action="store_true", help="run the small end-to-end smoke configuration")
    parser.add_argument("--skip-report", action="store_true", help="do not invoke build_report.py")
    args = parser.parse_args()
    config = ExperimentConfig()
    loaders = build_loaders(config, quick=args.quick)
    map_model, _ = run_map_training(config, loaders, quick=args.quick)
    suffix = "_quick" if args.quick else ""
    checkpoint = Path(config.results_dir) / f"map_checkpoint{suffix}.pt"
    members, _ = run_abnn_training(config, loaders, checkpoint, quick=args.quick)
    evaluate_methods(config, loaders, map_model, members, quick=args.quick)
    make_all_figures(config.results_dir, "figures", quick=args.quick)
    if not args.skip_report:
        from build_report import build_report
        build_report(quick=args.quick)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests and the actual quick workflow**

Run:

```bash
python -m pytest -q
python -m code.run_all --quick --skip-report
```

Expected: all tests pass; quick checkpoints, JSON, and six figures exist without a traceback.

- [ ] **Step 5: Commit**

```bash
git add "AI Assignment2/code/run_all.py" "AI Assignment2/tests/test_run_all.py" "AI Assignment2/results" "AI Assignment2/figures"
git commit -m "feat: add end-to-end ABNN experiment runner"
```

### Task 10: Word/PDF assignment report

**Files:**
- Create: `build_report.py`
- Test: `tests/test_report.py`
- Create: `report/AI_Assignment2_ABNN_Report.docx`
- Create: `report/AI_Assignment2_ABNN_Report.pdf`

**Interfaces:**
- Produces: `build_report(quick: bool = False) -> tuple[Path, Path]`.
- Reads JSON and PNG artifacts; it never embeds invented experiment values.

- [ ] **Step 1: Write a failing document-content test**

```python
from docx import Document
from build_report import build_report


def test_quick_report_has_required_sections(tmp_path, monkeypatch):
    docx_path, pdf_path = build_report(quick=True)
    text = "\n".join(p.text for p in Document(docx_path).paragraphs)
    for heading in (
        "1. How I Asked AI Tools to Find the Algorithm",
        "2. Algorithm Description",
        "3. How AI Implemented the Algorithm",
        "4. Experiment Settings and Results",
        "5. What I Have Learnt",
        "6. Webpage Link of the Source Codes",
    ):
        assert heading in text
    assert "HOI KAI CHENG" in text
    assert "UC325381" in text
    assert pdf_path.exists() and pdf_path.stat().st_size > 10_000
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_report.py -v`

Expected: FAIL because `build_report.py` does not exist.

- [ ] **Step 3: Implement data-driven DOCX composition**

`build_report.py` must:

```python
ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://github.com/HOIKAICHENG/CISC3024/tree/main/AI%20Assignment2"
STUDENT_NAME = "HOI KAI CHENG"
STUDENT_ID = "UC325381"
```

Create a title page and the six exact sections from the test. Include:

- the original user prompt and the three AI-researched candidate papers;
- CVPR 2024 citation and equations for MAP, Bayesian prediction, BNL, and finite Monte Carlo averaging;
- the AI implementation sequence and short, real code excerpts read from source files;
- a settings table and metric tables loaded from `evaluation.json`;
- all six generated figures with captions;
- limitations covering CPU model scale, short schedule, one seed, and OOD dataset choice;
- a first-person learning reflection;
- the clickable `SOURCE_URL`.

Use Word styles, numbered headings, page numbers, figure/table captions, and references. Convert DOCX to PDF with `win32com.client.DispatchEx("Word.Application")`, always closing the document and calling `word.Quit()` in `finally`.

- [ ] **Step 4: Generate and inspect the quick report**

Run:

```bash
python build_report.py --quick
python -m pytest tests/test_report.py -v
```

Expected: the report test passes and both quick report files open successfully.

- [ ] **Step 5: Commit**

```bash
git add "AI Assignment2/build_report.py" "AI Assignment2/tests/test_report.py"
git commit -m "feat: generate ABNN assignment report"
```

### Task 11: Full experiment, final report, reproducibility documentation, and publication

**Files:**
- Create: `README.md`
- Modify: `report/AI_Assignment2_ABNN_Report.docx`
- Modify: `report/AI_Assignment2_ABNN_Report.pdf`
- Create/Modify: `results/*.json`
- Create/Modify: `figures/*.png`
- Include: `AIassign2.pdf`

**Interfaces:**
- Final public URL: `https://github.com/HOIKAICHENG/CISC3024/tree/main/AI%20Assignment2`.

- [ ] **Step 1: Run the full experiment with close monitoring**

Run: `python -m code.run_all --skip-report`

Expected: 20 MAP epochs and 5 epochs for each of three members finish; `results/evaluation.json` and six final figures are written. If interrupted, rerun the same command and resume from the last checkpoint.

- [ ] **Step 2: Validate result integrity before report generation**

Run:

```bash
python -c "import json, pathlib; p=json.loads(pathlib.Path('results/evaluation.json').read_text()); assert set(p['methods'])=={'map','abnn_single','abnn_full'}; assert all(0<=m['classification']['accuracy']<=1 for m in p['methods'].values()); print({k:v['classification'] for k,v in p['methods'].items()})"
python -m pytest -q
```

Expected: method metrics print, all accuracy values lie in `[0, 1]`, and the full test suite passes.

- [ ] **Step 3: Generate the final DOCX and PDF**

Run: `python build_report.py`

Expected: `report/AI_Assignment2_ABNN_Report.docx` and `.pdf` exist and contain final rather than quick metrics.

- [ ] **Step 4: Write the reproduction README**

The README must include:

```markdown
# ABNN on CIFAR-10 — CISC3024 AI Assignment #2

AI-generated PyTorch reproduction of “Make Me a BNN” (CVPR 2024), adapted to
ResNet-20 and CPU execution. It compares deterministic MAP prediction with
single-member and three-member ABNN uncertainty estimates on CIFAR-10, SVHN,
and CIFAR-100.

## Reproduce

pip install -r requirements.txt
python -m pytest -q
python -m code.run_all --quick
python -m code.run_all
python build_report.py

## Source

https://github.com/HOIKAICHENG/CISC3024/tree/main/AI%20Assignment2
```

Also document the directory layout, measured runtime, actual final metric summary, method adaptations, and the CVPR paper citation.

- [ ] **Step 5: Inspect final repository status and commit all assignment artifacts**

Run:

```bash
git status --short
git diff --check
git add "AI Assignment2"
git commit -m "Add ABNN probability and Bayes assignment"
```

Expected: no whitespace errors; code, JSON logs, figures, report, README, and assignment PDF are committed while downloaded datasets and model checkpoints remain ignored.

- [ ] **Step 6: Verify the commit and push**

Run:

```bash
git status --short
git log -3 --oneline
git push origin main
```

Expected: clean working tree and a successful update of `origin/main`.

- [ ] **Step 7: Verify the public source page**

Open `https://github.com/HOIKAICHENG/CISC3024/tree/main/AI%20Assignment2`.

Expected: README, source files, JSON results, figures, assignment brief, and both report formats are visible; the same URL appears in the report.


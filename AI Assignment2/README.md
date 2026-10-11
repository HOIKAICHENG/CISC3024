# ABNN on CIFAR-10 — CISC3024 AI Assignment #2

AI-generated PyTorch reproduction of “Make Me a BNN” (CVPR 2024), adapted to
ResNet-20 and CPU execution. It compares deterministic MAP prediction with
single-member and three-member ABNN uncertainty estimates on CIFAR-10, SVHN,
and CIFAR-100.

## Reproduce

```text
pip install -r requirements.txt
python -m pytest -q
python -m code.run_all --quick
python -m code.run_all
python build_report.py
```

The experiment is deterministic for the recorded seeds and resumes compatible
MAP and ABNN checkpoints after interruption. Downloaded datasets and checkpoint
files are intentionally ignored by Git. The full command uses all 50,000
CIFAR-10 training images, 20 MAP epochs, three ABNN members with five adaptation
epochs each, and three stochastic inference draws per member.

## Measured full-run results

The approved run completed on Windows CPU in **7,536.411 seconds
(2 h 5 min 36 s)**. The persisted MAP training portion took 2,296.27 seconds.
The values below were loaded from `results/evaluation.json`; no settings or
values were tuned after observing them.

| Method | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| MAP | 0.8892 | 0.3298 | 0.1595 | 0.0219 |
| ABNN single member | 0.1570 | 2.4752 | 0.9483 | 0.1878 |
| ABNN three members (3×3 inference) | 0.2036 | 2.1437 | 0.8683 | 0.0360 |

| Method | SVHN AUROC / AUPR / FPR95 | CIFAR-100 AUROC / AUPR / FPR95 |
| --- | --- | --- |
| MAP | 0.8979 / 0.9371 / 0.3138 | 0.8461 / 0.8116 / 0.4557 |
| ABNN single member | 0.5065 / 0.7036 / 0.8779 | 0.5298 / 0.5052 / 0.9161 |
| ABNN three members | 0.3021 / 0.6116 / 0.9842 | 0.4987 / 0.5127 / 0.9546 |

The ABNN variants performed substantially worse than MAP in this run, including
near-random CIFAR-10 accuracy and weaker entropy-based OOD detection. These
surprising measurements are retained rather than hidden or tuned away. They
show that this short, small-model adaptation does not reproduce the paper's
large-model performance; one seed is insufficient for a statistical claim.

## Method adaptations

- CIFAR ResNet-20 replaces the paper's larger pre-trained architectures.
- MAP uses SGD, one warm-up epoch, cosine decay, and seed 0.
- Bayesian normalization perturbs BatchNorm scales; only normalization affine
  parameters are adapted.
- Member seeds 101, 102, and 103 emphasize three seeded classes at weight 5 to
  encourage different local modes.
- CIFAR-10 is in-distribution; SVHN and CIFAR-100 are OOD datasets. Predictive
  entropy is the OOD score.
- Evaluation compares deterministic MAP, one ABNN member with three draws, and
  three members with three draws each.

## Repository layout

- `code/`: configuration, data, model, ABNN conversion, training, evaluation,
  metrics, figure generation, and orchestration.
- `tests/`: unit and integration tests.
- `results/`: persisted quick/full JSON histories and evaluations.
- `figures/`: six final figures plus their SHA-256 provenance manifest.
- `report/`: final Word and PDF report.
- `build_report.py`: data-driven DOCX composition and Word PDF export.
- `AIassign2.pdf`: original assignment brief.

`results/evaluation.json` records the effective full configuration and
checkpoint SHA-256 provenance. `figures/manifest.json` binds all six PNG files
to the exact evaluation and history JSON files used to generate them.

## Reference

G. Franchi, O. Laurent, M. Leguéry, A. Bursuc, A. Pilzer, and A. Yao,
“Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from
Pre-trained Models,” *Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR)*, 2024, pp. 12194–12203.

https://openaccess.thecvf.com/content/CVPR2024/html/Franchi_Make_Me_a_BNN_A_Simple_Strategy_for_Estimating_Bayesian_CVPR_2024_paper.html

## Source

https://github.com/HOIKAICHENG/CISC3024/tree/main/AI%20Assignment2

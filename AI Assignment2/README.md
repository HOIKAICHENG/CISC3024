# ABNN on CIFAR-10 — CISC3024 AI Assignment #2

A from-scratch PyTorch re-implementation of **Adaptable Bayesian Neural Networks
(ABNN)** — Franchi et al., *Make Me a BNN: A Simple Strategy for Estimating
Bayesian Uncertainty from Pre-trained Models*, **CVPR 2024** — scaled down to a
laptop CPU with CIFAR-10 (in-distribution) and SVHN (out-of-distribution).

Everything in this folder (code, experiments, figures and the report) was
produced by an AI coding agent (Cursor) as required by the assignment.

## What is reproduced

| Paper claim | Experiment here |
|---|---|
| Replace BatchNorm with a Bayesian Normalization Layer (BNL) | `bnl.py`: $\gamma(1+\epsilon)$, $\epsilon\sim\mathcal{N}(0,0.1^{2})$ |
| Fine-tune only $\gamma,\beta$ from a MAP checkpoint | `train_abnn.py`, 3 members × 5 epochs |
| Class-dependent loss weights for ensemble diversity | factor 5 on 4 random classes per member |
| Predictive posterior ≈ average of $M\times L$ samples | $M=3$ members, $L=3$ noise samples |
| Better calibration / OOD than a single softmax model | Acc, NLL, ECE, CIFAR-10 vs SVHN AUROC / AUPR / FPR95 |

## Layout

```
code/
  bnl.py           BNL and BN → BNL conversion
  resnet.py        CIFAR ResNet-20
  data.py          CIFAR-10 + SVHN loaders
  prepare_data.py  HuggingFace CIFAR-10 mirror + official SVHN test mat
  train_map.py     deterministic MAP training
  train_abnn.py    post-hoc ABNN fine-tuning
  evaluate.py      Acc / NLL / ECE / OOD
  make_figures.py  report figures
  run_all.py       one-command reproduction
build_report.py    generates report/*.docx and *.pdf from results/*.json
results/           JSON logs (+ checkpoints, not committed)
figures/           figures used in the report
report/            the assignment report (DOCX + PDF)
```

## Reproduce

```bash
pip install -r requirements.txt
python code/run_all.py            # ~1 h on a 20-core CPU
python code/run_all.py --quick    # smoke test only
python build_report.py            # Windows + MS Word required for the PDF step
```

Adaptations w.r.t. the paper: ResNet-20 instead of ResNet-50 / WideResNet-28×10,
20 MAP epochs and 5 ABNN epochs instead of the paper's long ImageNet-style
schedules, $M=L=3$ as in Table 1 of the paper.

## Reference

G. Franchi, O. Laurent, M. Leguery, A. Bursuc, A. Pilzer, A. Yao.
*Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models.*
CVPR 2024. arXiv:2312.15297.

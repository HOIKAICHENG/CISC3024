# ConvNeXt V2 + FCMAE on CIFAR-10 — CISC3024 AI Assignment #1

A from-scratch PyTorch re-implementation of **ConvNeXt V2** (Woo et al., CVPR 2023) —
the *Global Response Normalization* (GRN) block and the *Fully Convolutional Masked
AutoEncoder* (FCMAE) pre-training — scaled down to run on a laptop CPU with CIFAR-10.

Everything in this repository (code, experiments, figures and the report) was produced by an
AI coding agent (Cursor, Claude) as required by the assignment; see the report for the prompts.

## What is reproduced

| Paper claim | Experiment here |
|---|---|
| FCMAE self-supervised pre-training of a ConvNeXt | `pretrain.py`, 50k unlabeled CIFAR-10 images, 60 % masking |
| ConvNeXt V1 + MAE suffers from *feature collapse*; GRN fixes it | `analysis.py`: channel cosine-distance per block, V1 vs V2 |
| GRN and FCMAE are complementary | 2 × 2 study {V1, V2} × {scratch, FCMAE init} on 5k labels (`finetune.py`) |

## Layout

```
code/
  convnextv2.py    GRN, ConvNeXt V1/V2 block, 4-stage backbone (Atto config)
  fcmae.py         masking, masked encoder, mask tokens, 1-block decoder, normalised-pixel loss
  data.py          CIFAR-10 loaders (stratified 5k labelled subset)
  prepare_data.py  fetch CIFAR-10 from the HuggingFace mirror and repack for torchvision
  pretrain.py      FCMAE pre-training
  finetune.py      supervised training (from scratch or from an FCMAE checkpoint)
  engine.py        cosine LR, param groups, evaluation
  analysis.py      feature-collapse diagnostic, k-NN on frozen features, reconstructions
  make_figures.py  all report figures
  run_all.py       one-command reproduction
build_report.py    generates report/*.docx and *.pdf from results/*.json
results/           JSON logs of every run (+ checkpoints, not committed)
figures/           figures used in the report
report/            the assignment report (DOCX + PDF)
```

## Reproduce

```bash
pip install -r requirements.txt
python code/run_all.py            # ~1.5 h on a 20-core CPU
python code/run_all.py --quick    # a few minutes, smoke test only
python build_report.py            # Windows + MS Word required for the PDF step
```

Adaptations w.r.t. the paper (all documented in the report): 32×32 inputs with a stride-2 stem,
mask grid decoupled from the encoder stride (4×4-pixel patches), dense masking instead of sparse
convolutions, decoder width 128, 8 pre-training / 30 fine-tuning epochs.

## Reference

S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, S. Xie. *ConvNeXt V2: Co-designing and
Scaling ConvNets with Masked Autoencoders.* CVPR 2023. arXiv:2301.00808.
Official code: https://github.com/facebookresearch/ConvNeXt-V2

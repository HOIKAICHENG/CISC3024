# ABNN AI Assignment Design

## Goal

Produce a complete, reproducible CISC3024 AI Assignment #2 submission about
Adaptable Bayesian Neural Networks (ABNN), published at CVPR 2024. The AI agent
will create all source code, run the experiments, generate a Word/PDF report,
and publish the finished work in the existing public repository:

https://github.com/HOIKAICHENG/CISC3024

The report identifies the student as HOI KAI CHENG (UC325381) and addresses all
six items in the assignment brief.

## Selected Approach

Implement the ABNN method directly in PyTorch and evaluate it on CIFAR-10. This
approach was chosen over a small MNIST study because CIFAR-10 is a more
convincing computer-vision benchmark, and over a TorchUncertainty-based
implementation because direct code makes the algorithm and the AI's
implementation easier to explain and verify.

The implementation is a CPU-scale reproduction rather than an attempt to match
the paper's ResNet-50 and WideResNet-28x10 results. It preserves the method's
central mechanism:

1. Train a deterministic maximum-a-posteriori (MAP) classifier.
2. Convert each BatchNorm layer to a Bayesian Normalization Layer (BNL).
3. Freeze the backbone and fine-tune only normalization parameters in several
   ABNN members.
4. Draw Gaussian perturbations from every member at inference and average their
   class probabilities.

## Model and Bayesian Layer

The deterministic backbone is a CIFAR-style ResNet-20 with BatchNorm after each
convolution. It is small enough for the approved one-to-two-hour CPU budget but
retains the residual and normalization structure required by ABNN.

For normalized activation `z`, the Bayesian normalization operation is:

`BNL(z) = z_hat * gamma * (1 + epsilon) + beta`,

where `epsilon ~ Normal(0, 1)` is sampled independently for each channel and
forward pass. Conversion copies the source BatchNorm running mean, running
variance, scale, bias, and batch counter. Only normalization affine parameters
are trainable during ABNN adaptation; convolutional and classifier parameters
remain frozen.

Three ABNN members start from the same MAP checkpoint. Each member prioritizes a
different seeded subset of classes with a class-weighted cross-entropy loss,
following the paper's mode-diversity mechanism. Inference averages three
Gaussian samples from each of the three members. The evaluated methods are:

- deterministic MAP checkpoint;
- one adapted ABNN member with three stochastic samples;
- full ABNN with three members and three samples per member.

## Data and Experiment Settings

- In-distribution training/test data: CIFAR-10, 50,000/10,000 images.
- Out-of-distribution data: SVHN test and CIFAR-100 test.
- Input size: 32 by 32 RGB.
- MAP training: 20 epochs, batch size 128, SGD with momentum, weight decay,
  warm-up followed by cosine learning-rate decay, random crop, and horizontal
  flip.
- ABNN adaptation: three members, five epochs per member, normalization
  parameters only, deterministic per-member class emphasis, and no OOD training
  data.
- Reproducibility: fixed Python, NumPy, and PyTorch seeds; settings stored with
  every result file.

The primary in-distribution metrics are accuracy, negative log-likelihood,
Brier score, and expected calibration error. OOD detection uses predictive
entropy and reports AUROC, AUPR, and FPR95 for each OOD dataset. The report will
avoid interpreting small differences as conclusive because the reproduction
uses one training seed and a short schedule.

## Components and Data Flow

The implementation will contain focused modules:

- `code/data.py`: deterministic CIFAR-10, CIFAR-100, and SVHN loaders.
- `code/model.py`: CIFAR ResNet-20 and feature/prediction interfaces.
- `code/abnn.py`: BNL, model conversion, parameter freezing, class-diverse
  member adaptation, and Monte Carlo prediction.
- `code/metrics.py`: classification, calibration, and OOD metrics.
- `code/train_map.py`: deterministic checkpoint training and JSON logging.
- `code/train_abnn.py`: member creation, adaptation, and checkpoint logging.
- `code/evaluate.py`: MAP/single-member/full-ABNN evaluation.
- `code/make_figures.py`: all report figures from persisted result files.
- `code/run_all.py`: full and quick end-to-end workflows with resume support.
- `tests/`: unit and smoke tests for the probabilistic mechanism and metrics.
- `build_report.py`: DOCX generation and PDF conversion.

Training writes checkpoints and append-safe JSON logs. Evaluation reads those
checkpoints and writes a single structured result file. Figure and report
generation read only persisted outputs so every published number is traceable
to an experiment artifact.

## Figures and Report

Figures will include:

1. ABNN conversion and inference architecture.
2. MAP and ABNN adaptation learning curves.
3. Reliability diagrams and confidence histograms.
4. OOD predictive-entropy distributions.
5. Metric comparison across MAP, one-member ABNN, and full ABNN.
6. Most- and least-uncertain CIFAR-10 examples.

The generated report will contain:

1. The original user prompt and the AI's literature-search/selection process,
   including the ABNN, DDU, and I-EDL candidates.
2. Bayesian prediction, MAP estimation, BNL, multimodal members, stochastic
   inference, and class-diversity loss.
3. The implementation workflow, file layout, key code excerpts, tests, and
   CPU-specific adaptations.
4. Exact experiment settings, tables, figures, measured results, and honest
   limitations.
5. A first-person reflection on Bayesian uncertainty, AI-assisted development,
   validation, reproducibility, and compute constraints.
6. A direct webpage link to the `AI Assignment2` folder in the GitHub
   repository.

Paper-reported results and assignment results will be labeled separately. The
report will not claim that a short, single-seed CPU experiment reproduces the
paper's absolute numbers.

## Verification and Failure Handling

Unit tests will verify:

- BNL is deterministic when stochastic sampling is disabled and stochastic
  when enabled.
- BatchNorm state and affine values survive conversion.
- only intended normalization parameters are trainable after freezing.
- all predictive probability vectors are finite and sum to one.
- ECE, Brier, AUROC, AUPR, and FPR95 match small hand-checkable cases.

A quick workflow will train on a small subset and execute conversion,
adaptation, evaluation, plotting, and report generation before the full run.
Dataset-download and checkpoint errors will fail with actionable messages.
Epoch-level checkpoints and logs permit safe resumption.

Completion requires passing unit tests, a passing quick workflow, successful
full experiments, internally consistent result/figure/report artifacts, valid
DOCX and PDF output, and a successful push to the configured GitHub remote.


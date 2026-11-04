# RotatE training in Google Colab

## Recorded completed run

The supplied Colab-budget handoff completed 30,000 updates on a Tesla T4.
Update 30,000 achieved the best full-validation expected random MRR of
0.2863605777207394. This is validation checkpoint selection. The final
40,932-query test audit remains with the shared evaluator.

Read the [experiment report](../reports/ROTATE_REPORT.md) for the verified
run, environment, learning curve, and limitations. Use the
[evaluation protocol](EVALUATION.md) for the final test audit.

## What is implemented

- RotatE's scorer is taken from the re-evaluation authors' pinned implementation.
- Training uses its self-adversarial negative-sampling objective, subsampling
  weights, alternating head/tail batches, Adam, and learning-rate decay.
- Only train triples supply positive examples and negative-sampling exclusions.
- Weights are selected by analytically expected random filtered MRR on validation.
- Other known true triples from train + valid + test are used only as a filter.
- Validation runs on all entities in both directions. Both real training presets
  use the full validation split; the smoke test uses eight validation triples.
- Checkpoints include sampler permutations/positions, optimizer, and random
  states so an interrupted run can resume from its latest saved update.
- Training and scoring use float32 without mixed precision or TF32.
- The final package is designed for the shared evaluator. No final test metrics
  are computed by the training script.

## Colab notebook

Upload `notebooks/RotatE_Colab.ipynb` at
[Google Colab](https://colab.research.google.com/) using **File > Upload notebook**.
Choose **Runtime > Change runtime type > T4 GPU**, then run the cells in order.
The notebook embeds this project's source code; it does not need a GitHub push.

The notebook will:

1. Restore the embedded source package.
2. Check that a CUDA GPU is available and record the environment.
3. Download and verify FB15k-237 using the shared data script.
4. Run the scorer, gradient, sampler, mapping, and filtering tests.
5. Perform a very short smoke training run to check the full pipeline.
6. Optionally mount your Google Drive and select a run directory.
7. Train the selected real configuration, resuming if a checkpoint exists.
8. Verify and export the validation-selected checkpoint for the evaluator.
9. Download the evaluator ZIP and its archive checksum.

Google Drive access, sign-in prompts, and runtime selection must be completed
by the user. Drive is recommended so saved checkpoints survive disconnections.
If Drive is disabled, outputs live under `/content` and can disappear when the
runtime ends; download them before disconnecting.

## Choose the training preset explicitly

| Setting | `colab_budget` (default) | `paper` |
| --- | ---: | ---: |
| Complex embedding dimension | 200 | 1000 |
| Gamma | 12 | 24 |
| Effective batch size | 512 | 1024 |
| Microbatch size | 64 | 32 |
| Negative samples per positive | 128 | 256 |
| Learning rate | 0.0001 | 0.0001 |
| Adversarial temperature | 1 | 1 |
| Planned optimizer updates | 30,000 | 150,000 |
| Validation interval | 3,000 | 10,000 |
| Validation triples | All 17,535 | All 17,535 |
| Training seed | 42 | 42 |

The default is a documented compute-limited checkpoint suitable for an initial
audit. It cannot be used to claim exact numerical reproduction of Table 2.
The `paper` preset copies the listed hyperparameters from the upstream RotatE
README, but it requires substantially more compute. Its source example has a
misleading `data/FB15k` path; this project uses verified FB15k-237 explicitly.

Microbatching reduces GPU memory by accumulating gradients over smaller pieces
of the effective batch. Subsampling weights are normalized across the entire
effective batch. Tests compare the loss and optimizer update with upstream
training. Floating-point accumulation order can still differ from a single
large batch.

These are seeded new training runs, not the paper's original weights. Training
time depends on the assigned GPU and preset. The log prints a measured
training-time estimate after the first 100 updates; it excludes validation,
checkpoint transfer, and export overhead. No fixed completion time is promised.

## Local commands

Install the bootstrap dependencies, then run data preparation and tests:

```sh
python -m pip install -r requirements-rotate.txt
python prepare_data.py
python -m unittest discover -s tests -v
```

Start the real Colab-budget run on a CUDA machine:

```sh
python -m rotate.train --config configs/rotate_colab_budget.json --output runs/rotate_colab_budget --device cuda
```

Resume from the latest atomic checkpoint with the same source and configuration:

```sh
python -m rotate.train --config configs/rotate_colab_budget.json --output runs/rotate_colab_budget --device cuda --resume
```

Use `configs/rotate_paper.json` and a separate output directory for the paper
preset. A changed seed, model size, sampler setting, or source requires a new
run. Deterministic resume is intended for the same software/device environment;
different GPU types or library versions can change float32 arithmetic.

CPU plumbing check only:

```sh
python -m rotate.train --config configs/rotate_smoke.json --output runs/rotate_smoke --device cpu
python -m rotate.verify --checkpoint runs/rotate_smoke/best.pt --report reports/rotate_smoke_adapter_checks.json
```

Never report smoke scores as the project's experiment.

## Outputs

Each run directory contains:

- `best.pt`: lightweight weights selected using validation MRR, including a
  step-zero baseline if training has not yet beaten initialization.
- `latest.pt`: model, optimizer, sampler, and RNG state for resuming training.
- `config.json`, `environment.json`, `requirements-lock.txt`, and
  `source_provenance.json`: configuration, exact direct package versions,
  platform/CUDA/GPU information, and source hashes.
- `training.jsonl` and `validation.jsonl`: training and selection history.
- `validation_indices.json`: exact validation rows used for selection.
- `summary.json`: completed/planned updates, selected step, validation metrics,
  timing, observed process memory, and GPU memory peaks.

`requirements-lock.txt` records the versions actually used, rather than forcing
Colab to replace its CUDA-enabled PyTorch with a local CPU build. It pins the
direct dependencies; platform, Python, and CUDA are recorded separately.

## Evaluator handoff

After training completes:

```sh
python -m rotate.export --run runs/rotate_colab_budget --output artifacts/rotate_handoff --device cpu
```

The command verifies the adapter and creates a ZIP plus a `.zip.sha256` file.
Share those two artifacts with the evaluator. See
[the handoff contract](ROTATE_HANDOFF.md) for the interface and checks.

If a deliberately shortened run is needed for midway, `--allow-partial`
permits exporting it with `training_complete: false`. The output must be
labelled preliminary. The exporter rejects smoke weights unless `--allow-smoke`
is explicitly supplied, and rejects an untrained selected checkpoint for a
real handoff.

## Compatibility changes and contribution

The upstream code is attributed in `vendor/kg_reeval/SOURCE.json`, with its
Apache-2.0 license included. Changes are limited to relative imports, removing
an unused scikit-learn import, and replacing removed NumPy `in1d` with `isin`.
The upstream RotatE scorer and `train_step` bodies are unchanged and checked
against recorded syntax-tree hashes from the original source.

The upstream `test_step` routine has a target-index/permutation issue and is
not used. Our additions are the verified data interface, resumable training
runner, memory-bounded gradient accumulation, validation-only selection,
raw-score adapter, checks, and evaluator package.

No GitHub push is performed by any training/notebook/export command.

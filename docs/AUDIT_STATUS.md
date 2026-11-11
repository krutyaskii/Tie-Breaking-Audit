# Run the frozen-checkpoint audit

The evaluator supports RotatE, corrected ConvKB (`convkb_hrt_v1`) and explicit
legacy diagnostics. Both corrected checkpoints are available. Local synthetic
tests and CPU validation pilots are complete. A full two-model test run is
still required; validation results are not test results.

## Environment and input packages

Tested: Python 3.13.15, torch 2.11.0+cpu, NumPy 2.3.3, matplotlib 3.10.7,
psutil 7.2.2. `requirements-evaluation.txt` pins these package versions. The
evidence bundle records the actual torch build, platform and numerical backend.
The GPU notebook is prepared; its execution requires an actual GPU runtime.

Obtain the separately shared ZIPs (weights and datasets stay out of Git):

| Package | ZIP SHA-256 |
| --- | --- |
| RotatE 20261102_130712 | `2c66eecac4fe499f8c3d695cb107d1c6025cda6758c875ea0461ba59421a0d1b` |
| Corrected ConvKB 20261111_025759 | `9008666f453d8306531b6cbf884cbd79b22d380198ba5124a205e47745065a5b` |

ConvKB selects step 6000, training seed 42, after 30000 completed updates.
Its validation MRR is 0.15885317359985426. The older 20261108 package has
the legacy architecture and cannot substitute for corrected weights.

## One-command evaluation

Run from the repository root, replacing ZIP paths:

```sh
python -m pip install -r requirements-evaluation.txt
python audit_pipeline.py --rotate-archive /path/rotate.zip --convkb-archive /path/convkb.zip --out runs/final_audit --device cuda
```

The runner verifies archive and internal hashes, checkpoint selection and
dictionaries, runs all tests, compares archived validation scores with
`atol=1e-4, rtol=1e-5`, checks repeatability in both modes, pilots both models,
then scores 40932 test queries per model. It imports repository code only.
Scores are unrounded float32; ties use exact equality, regardless of the
tolerance used for cross-device compatibility checks.

Fixed validation-pilot settings: RotatE batch 8/chunk 2048, ConvKB batch
2/chunk 512, four CPU threads, no AMP/TF32. CPU evaluation is supported but
the full ConvKB run can take hours. `--smoke-only` stops after validation:

```sh
python audit_pipeline.py --rotate-archive /path/rotate.zip --convkb-archive /path/convkb.zip --out runs/validation_check --device cpu --smoke-only
python audit_pipeline.py --rotate-archive /path/rotate.zip --convkb-archive /path/convkb.zip --out runs/final_audit --device cuda --resume
```

Resume rejects changed model/data/source/environment/device/batch/chunk
settings. State contains complete scoring batches, canonical query IDs and a
configuration hash. Progress is saved every 32 batches and at mode boundaries.
A crash can require recomputing unsaved batches; committed queries are not
duplicated. `audit.py run --stop-after-batches 1` allows controlled resume tests.

## Recompute and export without model scoring

```sh
python audit.py aggregate runs/final_audit/rotate
python audit.py aggregate runs/final_audit/convkb
python audit.py table runs/final_audit/rotate runs/final_audit/convkb --out runs/final_audit
python audit_bundle.py runs/final_audit
```

Aggregation needs NumPy/matplotlib and does not import torch. It checks file
and configuration hashes, canonical query order, target hashes, count bounds
and policy bounds. Outputs include metrics, seed results, exact tie histograms,
PNG distributions including singleton groups, tables and metadata. The ZIP
contains these outputs, test logs, environment lock and source, with SHA-256
sidecar, excluding weights, dataset splits and raw score tensors.

## Colab and optional retraining

Open `notebooks/Evaluation_Colab.ipynb`, select T4, upload both ZIPs and run
the cells. It embeds source and supports Google Drive resume without a GitHub
push. GPU execution remains to be validated on an actual GPU.

```sh
python -m convkb.train --config configs/convkb_smoke.json --output runs/convkb_training_smoke --device cpu
python -m convkb.train --config configs/convkb_colab_budget.json --output runs/convkb_retrain --device cuda
```

The supplied corrected checkpoint does not require retraining for this
checkpoint-specific audit. Its L2 applies to score_weight only. Historical
training metadata are preserved rather than rewritten.

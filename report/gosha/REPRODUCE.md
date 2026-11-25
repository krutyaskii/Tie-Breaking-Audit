# Reproduce Gosha's evaluation contribution

The report separates two different operations. Reconstructing saved metrics
does not repeat neural model scoring. The completed full GPU run is preserved
in `reports/final_audit`; a new full GPU run was not performed on 25 November.
All commands below are run from the repository root unless noted otherwise.

## A. Reconstruct results, tables and figures without weights or GPU

Requirements: Python 3.13 and the two pinned packages below. No PyTorch,
checkpoint archive, Google Drive access or dataset download is needed.

```sh
git clone https://github.com/krutyaskii/Tie-Breaking-Audit.git
cd Tie-Breaking-Audit
python -m venv .venv-report
```

Activate it (`.venv-report\Scripts\Activate.ps1` on Windows, or
`source .venv-report/bin/activate` on Linux/macOS), then:

```sh
python -m pip install -r report/gosha/requirements-report.txt
python report/gosha/generate.py --out runs/gosha_materials
```

This independently sums reciprocal ranks, checks all scopes/policies, draws
the five specified seeds, verifies sample standard deviations, histograms,
state/config hashes, canonical IDs and the saved JSON/CSV/Markdown tables.
Only after verification does it generate CSV/LaTeX tables, PDF/PNG figures,
`numbers.tex`, an input-hash manifest and `source_validation.json` in the
specified output directory. The frozen evidence is never overwritten.

To regenerate the committed materials in place, omit `--out`. The main text
loads `numbers.tex`; measured values are not typed manually into the prose.
PDF figure timestamps are disabled. Pixel rendering can still depend on the
platform/font stack; numerical checks, not image hashes, establish agreement.

For the existing evaluator's aggregation command, first copy the evidence:

```sh
python -c "import shutil; shutil.copytree('reports/final_audit', 'runs/recomputed')"
python audit.py aggregate runs/recomputed/rotate
python audit.py aggregate runs/recomputed/convkb
python audit.py table runs/recomputed/rotate runs/recomputed/convkb --out runs/recomputed
```

`audit.py aggregate` rewrites derived metrics/figures in its argument directory.
Copying first preserves the imported artifacts and their byte-level manifest.

## Benchmark-aware verification

To additionally verify targets and retained counts against the actual
train/valid/test union, reconstruct the dataset and validate the separately
provided original result ZIP:

```sh
python prepare_data.py
python tools/validate_evaluation_bundle.py /path/evaluation_validation_bundle.zip --out runs/independent_check
```

Original result ZIP SHA-256:
`88273c8dc37b838b9589cb3b6e43245535850d8125e47e121890477a8a481201`.
The validator reads the source snapshot as data and does not execute it.
The ZIP is distributed separately; the committed compact evidence suffices
for operation A even when that ZIP is unavailable.

## B. Repeat model scoring from frozen checkpoints

Install `requirements-evaluation.txt` in a separate environment with a working
CUDA-enabled PyTorch for a full GPU run. The checked CPU environment used
`torch==2.11.0+cpu`; CUDA was unavailable there. Actual package locks for both
fresh environments are under `logs/03_report_freeze.txt` and
`logs/12_evaluation_freeze.txt`. The original full run's actual CUDA build is
recorded in `reports/final_audit/*/metadata.json`.

Two exact archives must be obtained separately from the model owners:

| Model / package | Archive SHA-256 |
| --- | --- |
| RotatE `rotate_colab_budget_handoff_20261102_130712.zip` | `2c66eecac4fe499f8c3d695cb107d1c6025cda6758c875ea0461ba59421a0d1b` |
| Corrected ConvKB `convkb_colab_budget_handoff_20261111_025759.zip` | `9008666f453d8306531b6cbf884cbd79b22d380198ba5124a205e47745065a5b` |

A filename suffix such as `(1)` does not change archive identity; the hash
must match. The 8 November ConvKB archive is a legacy architecture and must
not replace the corrected 11 November package. No public download endpoint
for the weights has been verified. A Git clone alone therefore does not
suffice for operation B.

Validation-only smoke test (tested on CPU, 8 triples / 16 queries per model):

```sh
python audit_pipeline.py --rotate-archive /path/rotate.zip --convkb-archive /path/convkb.zip --out runs/checkpoint_smoke --device cpu --smoke-only
```

Full scoring command (provided; not newly executed on 25 November):

```sh
python audit_pipeline.py --rotate-archive /path/rotate.zip --convkb-archive /path/convkb.zip --out runs/final_repeat --device cuda
```

Resume by repeating the exact command with `--resume`. Model/data/source,
device/environment and batch/chunk settings must match. Each policy and seed
uses the same saved counts. Fixed settings are RotatE query batch 8/chunk
2048 and ConvKB batch 2/chunk 512; do not tune them using test results.
The self-contained `notebooks/Evaluation_Colab.ipynb` provides the GPU path.
Exact float32 ties are not guaranteed identical across devices or shapes.
No training command is needed or used for this contribution.

## Build the LaTeX previews

The official NeurIPS 2025 style file and its provenance are included. The
standard template is single-column. `preview.tex` is an explicitly adapted
two-column integration preview, as requested; it is not a conference-compliant
NeurIPS submission layout. `preview_neurips.tex` provides a single-column
alternative with the official style's typography and margins. The requested
two-column preview is two pages including references. The single-column
reference has two pages of contribution text and a references-only third page;
it is not a claim about the final combined document's page allocation.
Masha owns the final six-page document layout.

From `report/gosha`, using Tectonic 0.17.0 (tested):

```sh
tectonic --keep-logs --keep-intermediates preview.tex
tectonic --keep-logs --keep-intermediates preview_neurips.tex
```

The compiler downloads TeX packages/fonts on first use; its source URL,
official asset digest and style hashes are recorded in
`style_and_compiler_provenance.json`. It is not bundled as an executable in Git.
Rendering/inspection additionally used the optional pinned
`requirements-preview.txt`; these packages are not required for numerical
reconstruction. See `VALIDATION.md` for actual commands, exit codes and limits.

# Actual validation record — 25 November 2026

## Source and scope

A new GitHub clone was created at numerical/evidence base revision
`c08d07b111d86706991d6819216b1a68cbcf5c06`. Its initial working tree was clean.
Report files were then added under `report/gosha`; the evaluator, model code,
frozen count arrays and trained checkpoints were not modified. These checks
do not claim a newly repeated full GPU experiment.

`logs/commands.jsonl` records actual argument vectors, working directories,
exit codes, elapsed times and corresponding full text logs. Absolute paths
describe the validation machine; portable equivalents are in `REPRODUCE.md`.

## A. Saved-result reconstruction — PASS

Fresh environment A: Python 3.13.15, Windows 11, NumPy 2.3.3, matplotlib
3.10.7. PyTorch is **not installed** (`logs/02_report_environment.txt`).
PyMuPDF 1.26.6 is an optional PDF-inspection dependency, not a numerical
dependency. Full package freeze: `logs/03_report_freeze.txt`.

Executed checks, all exit 0:

- `report/gosha/generate.py`: all 32 imported artifact hashes; 40932 unique
  canonical row/mode IDs per model; target and state/config hashes; integer
  array shapes; B/T/retained bounds; direct reciprocal sums for overall/head/
  tail MRR; Hits@1/3/10 for all policies; policy bounds; exact T histograms
  and quantiles; five fresh default_rng draws; mean and sample std (ddof=1).
- Saved JSON, per-model and combined CSV, Markdown's stated rounding, seed
  CSVs and separate histogram JSON agree. Maximum direct-sum versus saved
  floating-point metric difference is **5.551115123125783e-17**, below the
  absolute verification tolerance 1e-13 (`source_validation.json`).
- `audit.py aggregate` and `audit.py table` were also run without PyTorch on
  copies under `runs/recomputed`, preserving original evidence bytes.
- Table/figure generation to a separate output directory was executed using
  the documented `--out runs/gosha_materials` option.

Logs: `31_generate_report*.txt`, `33_copy_saved_results.txt`,
`34_aggregate_*.txt`, `35_combined_table.txt`,
`36_regenerate_to_separate_directory.txt`.

After the report files were committed in candidate revision `4445d1d`, a
second, completely clean local Git clone of that commit was checked. It had
no prepared dataset, raw archive, run directory or PyTorch installation. The
documented requirements file and `generate.py --out runs/gosha_materials`
commands succeeded. All ten generated numeric/LaTeX files and all four figure
files matched the committed versions byte-for-byte in this environment;
all 32 frozen evidence hashes remained unchanged. The verification checkout
was clean both before and after generation. See
`checks/final_clean_clone.json` and `logs/60_*` through `logs/65_*`.
Later commits add this verification record only, without changing numerical
code or generated materials. Generated text files explicitly use LF endings
so checkout normalization does not break repeatability across these clones.

## Benchmark-aware verification — PASS

The fresh clone initially had no prepared dataset. `prepare_data.py` downloaded
the fixed source archive and verified original split bytes, IDs and hashes.
The first restricted-network attempt returned exit 1; the authorized network
retry returned exit 0. Both are recorded (`30_prepare_data*.txt`).

`tools/validate_evaluation_bundle.py` then independently verified the original
result ZIP and sidecar, all 74 internal hashes, all 28 bundled source files
against matching run code, canonical queries and every target/retained count
against the restored train/valid/test union. It independently recalculated
all policies, scopes, histograms and seeded statistics. Exit 0:
`logs/32_independent_bundle.txt`.

Source bundle SHA-256:
`88273c8dc37b838b9589cb3b6e43245535850d8125e47e121890477a8a481201`.

## B. Frozen-checkpoint scoring — CPU smoke PASS; new full GPU run BLOCKED

Fresh environment B: Python 3.13.15, torch 2.11.0+cpu, NumPy 2.3.3,
matplotlib 3.10.7, psutil 7.2.2. `torch.cuda.is_available()` is false
(`logs/11_evaluation_environment.txt`); full freeze is in
`logs/12_evaluation_freeze.txt`.

Both exact frozen ZIPs were available locally and verified before loading.
`audit_pipeline.py --device cpu --smoke-only` returned exit 0. Its internal
unit-test invocation passed **24 tests**; the full log is copied to
`logs/24_tests_from_smoke.txt`. This differs from the historical full Colab
run's 23 tests because the merged upstream formula test was added later.

The CPU pipeline verified packages, selected steps, dataset/dictionary hashes,
validation examples, repeatability in both modes and then scored only the
first eight validation triples (16 queries per model). Maximum differences
from archived CUDA scores were 3.814697265625e-6 for RotatE and
1.71661376953125e-5 for ConvKB, within the recorded compatibility tolerances
atol=1e-4, rtol=1e-5. Exact head/tail repeatability passed. The CPU check does
not claim identical cross-device ties and did not score test.

See `logs/41_cpu_checkpoint_smoke.txt` and `checks/cpu_*_adapter_checks.json`.

**BLOCKED:** a new complete CUDA scoring run requires a CUDA runtime (e.g.
Colab T4), the two exact archives listed in `REPRODUCE.md`, and the recorded
batch/chunk/precision settings. No local GPU is available. The full command
is provided but was not run in this validation session. The previously
completed full T4 run remains the source of the reported test results.

Weight availability: supplied local archives passed verification; a public
download endpoint was **not verified**. Do not describe a Git clone as
sufficient for full model-scoring reproduction. No new model training ran.

## LaTeX, figures and PDF — PASS

Tables and both English-labeled figures are generated from verified artifacts.
The target-tie figure uses a linear percentage axis and exact count/total;
the gap figure uses a logarithmic axis and retains tiny nonzero values. The
paper's unresolved unrounded RotatE gap is not plotted as an exact zero.

Tectonic 0.17.0 compiled both previews successfully, including BibTeX. Its
official Windows asset digest was verified; style/compiler provenance is in
`style_and_compiler_provenance.json`. The included style is the official
NeurIPS 2025 file. The two-column preview is an explicit course integration
adaptation, not a NeurIPS conference submission format.

`preview.pdf` has **two pages**, including references. The single-column
reference has two pages of main contribution plus one references-only page.
PDF inspection found no overfull boxes, unresolved reference markers or text
outside the checked margin tolerance, and working hyperlinks were present.
Both two-column pages were rendered and visually reviewed: tables, equations,
captions and the rare-tie labels are legible and within the margins.
See `pdf_validation.json`, `logs/50_latex_build*.txt`,
`logs/51_latex_single_column*.txt` and `logs/52_pdf_qa*.txt`.

## Source checks and team boundary

Course requirements and Table 2/Section 5.4 were checked directly on
25 November against the course page and ACL PDF. The paper's RANDOM is a
five-run mean, distinct from our analytic expectation. BibTeX metadata was
checked against ACL Anthology's official `.bib` record. Source links and
publication values are recorded in `paper_reference.json` and `references.bib`.

The contribution, short speaking text and AI-use wording are drafts for team
review. Final integration, the complete six-page PDF, shared sections, the
eight-minute deck and team approval are **PENDING / outside this contribution**.
No messages were sent to teammates and no approval was claimed.

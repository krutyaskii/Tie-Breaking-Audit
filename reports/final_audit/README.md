# Completed FB15k-237 frozen-checkpoint audit

Both full runs contain 40932 queries: 20466 head, then 20466 tail, original
test-file order. Mode 0 is head, mode 1 is tail. `T` includes the target;
`retained` includes the target and excludes other known train/valid/test truths.

See `main_table.md`, each model's `metrics.json`, and `validation_report.json`.
`per_query.npz` contains integer row/mode/target/B/T/retained arrays and an
embedded configuration hash. Model weights, dataset text and raw score arrays
are excluded. The compact counts are sufficient to reproduce all policies and
seeds without GPU scoring.

Run from the repository root:

```sh
python audit.py aggregate reports/final_audit/rotate
python audit.py aggregate reports/final_audit/convkb
python audit.py table reports/final_audit/rotate reports/final_audit/convkb --out reports/final_audit
```

`ARTIFACT_SHA256SUMS.json` records imported artifact bytes; aggregation can
regenerate derived files with platform-specific line endings or PNG metadata.
`SOURCE_BUNDLE.sha256` identifies the original returned Colab ZIP. It includes
the exact run source snapshot; source comparison permits only newline changes.

The independent validator uses the actual benchmark to check targets and
retained counts, direct sums to check metrics, and fresh seed draws to check
Monte Carlo summaries. It does not repeat GPU candidate scoring. The main
result is negligible tie impact on these budget checkpoints; the paper's large
ConvKB effect was not observed. See `../EVALUATION_REPORT.md` for limitations.

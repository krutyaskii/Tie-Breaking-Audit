# Evaluator status

The shared evaluator is under development. `audit.py` scores candidates in
small batches and retains per-query B/T counts. Filtering uses the union of
train/valid/test and always retains the target; ties are exact float32
equalities, using the target score in the same candidate row. Metrics include
optimistic, pessimistic and analytic random MRR/Hits, plus five random draws.

The supplied ConvKB checkpoint is a legacy architecture diagnostic, not a
corrected ConvKB model. Correct architecture and new validation-selected
weights are outstanding. Full two-model replication is blocked. Original
validation MRR values are not test results.

After reconstructing the data with `python prepare_data.py`, unpack the
separately supplied RotatE ZIP into `artifacts/rotate_bundle` and verify its
SHA256SUMS.json. With PyTorch and NumPy installed, run from the repository:

```
python audit.py run --model rotate --checkpoint artifacts/rotate_bundle/model.pt --data data/FB15k-237 --out runs/rotate_valid_smoke --split valid --limit 8 --batch 2 --chunk 2048 --device cpu
python audit.py aggregate runs/rotate_valid_smoke
```

The 8-triple sample is a smoke check, not a test result. To resume an
interrupted run, repeat its exact command with `--resume`; the evaluator checks
the recorded configuration and query prefix. `--split test` evaluates the
full test split and is not permitted with `--limit`. Fix device, batch and
chunk sizes after a validation memory pilot. Do not compare exact float32
ties across different devices as if they were numerically identical.

The evaluator is not yet a complete final audit: archival score agreement,
independent resume/fault-injection checks, reproducible full-run environment,
figures, combined tables and the full RotatE test remain to be completed.
Current verified local environment: Python 3.13.15, NumPy 2.3.3, torch
2.11.0+cpu; no CUDA device was available. This is not the original Colab
environment. The full test has not been run.

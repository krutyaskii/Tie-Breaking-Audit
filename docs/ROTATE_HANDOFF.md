# RotatE evaluator handoff

This bundle contains model weights selected using validation only, exact ID
mappings, raw-score adapter code, configuration, environment versions, training
logs, a data manifest, and a small validation score example. It contains no
dataset splits or final test metrics.

## Completed supplied run

The supplied `rotate_colab_budget_handoff_20261102_130712.zip` completed
30,000 updates with seed 42 and selected update 30,000 using full validation.
Its expected random MRR of 0.2863605777207394 is a **validation** result.
The archive records `test_queries_scored: 0`. Final evaluation uses all
20,466 test triples in both directions, giving 40,932 queries.

In the project repository, `reports/ROTATE_REPORT.md` covers measured results
and provenance, and `docs/EVALUATION.md` defines the final test formulas,
five evaluation seeds, and output acceptance criteria. These documents
accompany the original archive separately. The original archive remains
unchanged.

The current repository includes the protocol and adapter. The shared
evaluator's one-command test audit is a remaining team deliverable.

Read `handoff.json` first. It states the actual configuration, selected update,
whether training completed, and whether this is only a smoke example. A
`colab_budget` checkpoint is a compute-limited experiment, not a reproduction of
the paper's hyperparameters. Never describe smoke or unfinished weights as the
completed paper result.

## Open the bundle

Extract the ZIP into a new folder. Use a trusted Python environment with the
versions in `requirements-lock.txt`; the training machine and device are
recorded in `environment.json`. This lock lists the direct scoring/training
dependencies, while Python, CUDA, and hardware are recorded separately.

From the extracted folder:

```python
from pathlib import Path
import numpy as np
from rotate.adapter import RotatEAdapter

bundle = Path('.')
adapter = RotatEAdapter.load(bundle / 'model.pt', mapping_dir=bundle, device='cpu')
examples = np.load(bundle / 'validation_score_examples.npz')
triples = examples['triples']  # int64 rows in head, relation, tail ID order
head_scores = adapter.score_candidates(triples, mode='head')
tail_scores = adapter.score_candidates(triples, mode='tail')
print(head_scores.shape, head_scores.dtype)
```

For 8 queries, each output has shape `(8, 14541)` and dtype `float32`. Column
`j` is entity ID `j` in `entities.dict`. Larger scores mean more plausible
triples. No sigmoid, rounding, filtering, sorting, or tie-breaking is applied.

Batch queries using `query_batch_size` from `config.json`, or a smaller value
if memory is limited. Candidate chunking is handled internally using the saved
configuration. Score each checkpoint once on one recorded device and reuse
the resulting scores for every tie policy. Exact float32 ties can differ
between devices; do not mix CPU and GPU scores in one audit.

## Data and IDs

Use the shared FB15k-237 splits reconstructed with `python prepare_data.py`.
Original split hashes are in `data_manifest.json`. The adapter checks both
RotatE dictionary hashes before loading the model. The original source-format
dictionaries are also provided for comparison.

Do not reassign IDs. For each evaluated triple, score all entity candidates
for both head and tail prediction. Filter other known true answers using
train + valid + test, retaining the target. Define ties by exact equality of
the unrounded float32 scores returned by this adapter.

The supplied eight-query file is a validation-only integration example. It
is not a replacement for scoring the entire test split. Its stored scores
can have small numerical differences if you rerun on a different device.

## Verify the bundle

Run this before loading weights:

```python
import hashlib
import json
from pathlib import Path

root = Path('.')
reference = json.loads((root / 'SHA256SUMS.json').read_text())
for name, expected in reference.items():
    with (root / name).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == expected, name
print('All bundle checksums match')
```

The snippet uses Python 3.11 or newer. On Python 3.10, replace the
`hashlib.file_digest(...)` line with
`hashlib.sha256(stream.read()).hexdigest()`.

`adapter_checks.json` records checks against native upstream scoring for both
prediction modes, exact score repeatability on the verification device,
mapping hashes, and unchanged upstream scoring/training method bodies.

The outer `.zip.sha256` file verifies the archive itself. Keep it next to the
ZIP when transferring the bundle.

## Information to report

Report the preset, actual selected step, dataset hashes, checkpoint SHA-256,
training seed, environment/device, validation selection rule, training time,
observed memory use, and deviations from the paper. Results apply to this
checkpoint. Random tie-breaking seeds describe evaluation randomness at fixed
weights; they are not independent training runs.

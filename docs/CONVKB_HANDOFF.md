# ConvKB evaluator handoff

This bundle contains ConvKB weights selected using validation only, exact ID
mappings, raw-score adapter code, configuration, environment versions, training
logs, a data manifest, and a small validation score example. It contains no
dataset splits or final test metrics.

## Implementation note

This checkpoint is a **legacy ConvKB variant**, not a faithful ConvKB
implementation. Its convolution does not combine h/r/t at the same embedding
coordinate. Corrected architecture and retraining are deferred. Do not present
this checkpoint as a ConvKB replication.

The input is `[B,1,3,D]`; kernel `(1,3)` moves along D and processes h/r/t
separately. The legacy training objective uses `softplus(-pos) +
softplus(neg)` plus L2 on `score_weight` only. No claim of upstream objective
equivalence is made.

## Score direction

Larger scores mean more plausible triples. The adapter returns raw float32
scores without inversion. Candidate column `j` belongs to entity ID `j` in
`entities.dict`. No filtering, sorting, or tie-breaking is applied.

## Open the bundle

Extract the ZIP into a new folder. Use a trusted Python environment with the
versions in `requirements-lock.txt`; the training machine and device are
recorded in `environment.json`.

```python
from pathlib import Path
import numpy as np
from convkb.adapter import ConvKBAdapter

bundle = Path('.')
adapter = ConvKBAdapter.load(bundle / 'model.pt', mapping_dir=bundle, device='cpu')
examples = np.load(bundle / 'validation_score_examples.npz')
triples = examples['triples']
head_scores = adapter.score_candidates(triples, mode='head')
tail_scores = adapter.score_candidates(triples, mode='tail')
print(head_scores.shape, head_scores.dtype)
```
## Data and IDs

Use the shared FB15k-237 splits reconstructed with `python prepare_data.py`.
Original split hashes are in `data_manifest.json`. The adapter checks both
dictionary hashes before loading the model.

Do not reassign IDs. For each evaluated triple, score all entity candidates
for both head and tail prediction. Filter other known true answers using
train + valid + test, retaining the target. Define ties by exact equality of
the unrounded float32 scores returned by this adapter.

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
The supplied `adapter_checks.json` checks shape, finite scores, mapping and
head repeatability. It does not compare with an independent upstream scorer.

## Information to report

Report the preset, actual selected step, dataset hashes, checkpoint SHA-256,
training seed, environment/device, validation selection rule, training time,
observed memory use, and deviations from the paper. Results apply to this
checkpoint. Random tie-breaking seeds describe evaluation randomness at fixed
weights; they are not independent training runs.


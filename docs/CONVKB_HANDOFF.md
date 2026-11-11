# ConvKB evaluator handoff

The corrected 20261111_025759 package mixes h/r/t at each coordinate: input
`[B,1,3,D]`, kernel `(3,1)`, output `[B,F,1,D]`, ReLU and `F*D` readout.
The architecture identifier is `convkb_hrt_v1`. The adapter detects the
architecture from tensor shapes and loads weights strictly.

The 20261108 package used kernel `(1,3)` and `F*3*(D-2)` readout, processing
h/r/t separately. `convkb/legacy_model.py` preserves it for diagnostics with
explicit `allow_legacy=True` and identity `legacy_convkb_variant`.

## Corrected package evidence

- ZIP SHA-256: `9008666f453d8306531b6cbf884cbd79b22d380198ba5124a205e47745065a5b`.
- model.pt SHA-256: `dfa921d625ae229320106a6321844991a385ca4f9a3cb87117eb29e212cbfe46`.
- 40/40 internal checksums match; dictionaries match the shared FB15k-237 IDs.
- 30000 updates completed, seed 42, best step 6000 selected on 17535 validation
  triples; validation MRR 0.15885317359985426. Zero test queries in the package.

The local objective is `softplus(-positive)+softplus(negative)` plus L2 on
`score_weight` only; higher scores are better. The archive's docstring also
mentions L2 on conv.weight, which is inaccurate. This budget PyTorch recipe
does not establish equivalence to the original paper's TensorFlow training.

Verify the archive and SHA256SUMS.json before loading; preserve dictionary IDs:

```python
from convkb.adapter import ConvKBAdapter
adapter = ConvKBAdapter.load('package/model.pt', 'data/FB15k-237', device='cpu')
scores = adapter.score_candidates([[0, 0, 1]], mode='tail')
```

Scores are finite unrounded float32, column j is entity ID j. The adapter does
not filter or rank. See [AUDIT_STATUS.md](AUDIT_STATUS.md) for full commands.

Local verification checks repeatability in both modes and fails on false
results. Independent hand-set coordinate formula and mapping tests are in
`tests/test_convkb.py`. The archive verifier checks only head repeatability
and does not independently compare upstream architecture. Its historical
provenance does not cover ConvKB source; future local runs hash convkb/*.py.
Historical archive files are not rewritten.

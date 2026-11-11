# Frozen-checkpoint tie-breaking audit

## Question and protocol

How much can optimistic tie handling inflate MRR relative to uniform random
tie handling on fixed RotatE and ConvKB checkpoints? The primary effect is
MRR_optimistic minus analytic E[MRR_random], in absolute MRR units.
FB15k-237 has 14541 entities, 237 relations and 20466 test triples. A complete
audit has 20466 head queries followed by 20466 tail queries per model, in
original split order. Other known train/valid/test answers are filtered,
retaining the target. Scores are finite unrounded float32, and the target's
score comes from its own candidate row. Ties use exact equality.

B counts strictly better candidates; T counts equals including the target.
Ranks are B+1 (optimistic) and B+T (pessimistic). Expected random RR averages
reciprocals over that interval, rather than inverting mean rank. Five draws
use NumPy default_rng seeds 0–4 in canonical order, with sample standard
deviation ddof=1. They measure evaluation randomness at fixed weights.

## Checkpoints and architecture audit

RotatE selects step 30000, seed 42, validation MRR 0.2863605777207394. Corrected
ConvKB (20261111) selects step 6000 after 30000 updates, seed 42, validation
MRR 0.15885317359985426. Both packages record zero test queries. All archive,
internal file and checkpoint hashes, dictionaries and selection logs are
checked before scoring.

Corrected ConvKB mixes h/r/t at each coordinate with kernel (3,1) on
[B,1,3,D], then ReLU and F*D readout. A hand-set scalar-formula test checks
embeddings, convolution weights, biases and output weights independently.
The old checkpoint processes h/r/t separately and remains explicit legacy
diagnostics. Strict architecture-specific loading rejects incompatible weights.

Actual ConvKB L2 applies only to score_weight, despite the archive's contrary
docstring. Its loss and higher-is-better direction are checked locally without
changing trained weights. Historical provenance omits ConvKB source, and the
archive verifier checks repeatability only in head mode. Future local runs
hash ConvKB source and local verification checks both modes and fails on false
results. These fixes do not rewrite training history.

## Completed test experiment and independent validation

All 23 tests passed in the Colab run. They cover formulae, exact ties,
filtering, candidate IDs, independent architecture, invalid inputs, bounds,
seeded estimates and resume equivalence. Both models' head/tail compatibility
examples matched the archived CUDA examples exactly (maximum absolute error
zero); repeatability passed in both modes.

The full test audit completed on Tesla T4, Python 3.13.16, torch
2.11.0+cu130, CUDA 13.0, NumPy 2.3.3, matplotlib 3.10.7 and psutil 7.2.2.
RotatE used query batch 8/chunk 2048; ConvKB used batch 2/chunk 512. Both
used raw float32, TF32 disabled and deterministic algorithms enabled. RotatE
took 85.15 seconds and ConvKB 1263.78 seconds, excluding setup/pilots.

| Model | Optimistic MRR | Expected random MRR | Pessimistic MRR | Optimistic - expected |
| --- | ---: | ---: | ---: | ---: |
| RotatE | 0.282140899092 | 0.282140868651 | 0.282140838210 | 3.04409439322e-8 |
| Corrected ConvKB | 0.151383763468 | 0.151383763405 | 0.151383763342 | 6.30169250115e-11 |

Expected-random Hits@1/3/10 are 0.197547151373 / 0.312249584677 /
0.450820873644 for RotatE and 0.0756620736832 / 0.158409068699 /
0.312518323072 for ConvKB. Hits are identical for all three policies.

Exactly 26/40932 RotatE queries (0.063520%) and 3/40932 ConvKB queries
(0.007329%) have target ties. All these groups have T=2; all remaining queries
have T=1. Five seeded MRR draws have mean/sample-standard-deviation
0.282140879712 / 2.36135307794e-8 for RotatE and
0.151383763392 / 6.59143888951e-11 for ConvKB. The seeded-mean deviations from
analytic expectation are 1.10609710280e-8 and -1.34634525750e-11 respectively;
such Monte Carlo deviations are expected, not implementation failures.

The returned ZIP has SHA-256
`88273c8dc37b838b9589cb3b6e43245535850d8125e47e121890477a8a481201`.
All 74 internal hashes and its sidecar match. A separate validator reproduces
all scopes/policies by direct reciprocal summation, all five seeds, sample
standard deviations, exact histograms, quantiles and CSV tables. It verifies
40932 canonical query IDs per model, every target against the real test split,
every retained candidate count against union(train,valid,test), policy bounds,
checkpoint identity, and data/source/state hashes. All 28 bundled source files
match local code after newline normalization. See
[the independent report](final_audit/validation_report.json) and
[full tables](final_audit/main_table.md).

This post-run validation does not re-execute GPU candidate scoring: it verifies
the counts, protocol evidence and all downstream results. The bundled source,
checkpoint hashes, integration checks and logs document scoring provenance.

## Comparison and limitations

[Sun et al. (2020), Table 2](https://aclanthology.org/2020.acl-main.489.pdf)
reports FB15k-237 RotatE TOP/RANDOM/BOTTOM MRR 0.336/0.336/0.336, and ConvKB
0.407/0.243/0.130. Paper RANDOM averages five runs; our primary RANDOM is
analytic expectation, with five draws separately. Budget training, checkpoint
selection, PyTorch implementation and numerical environment differ. Matching
coordinate mixing does not prove full upstream recipe equivalence. This run
finds negligible tie-breaking impact for both checkpoints and does not reproduce
the paper's large ConvKB gap of 0.164. It is a completed checkpoint-specific
audit, not a replication of the paper's complete training recipe or a refutation
of its observations. The effect of individual recipe differences would require
controlled experiments. Precision, seeds, weights and training were not changed
in response to these test results.

## Contribution and AI use

Our additions are shared filtered ranking, analytic metrics, seeded estimates,
per-query evidence, resumable streaming with source/environment guards,
independent architecture tests, standalone aggregation, tables, exact and
binned histograms, validation export and independent returned-bundle checking.
RotatE uses vendored kg-reeval code
with its recorded revision/license. ConvKB retains its existing public-model
attribution; model training was performed separately.

OpenAI Codex assisted with code, tests, documentation and independent validation.
The team must review and explain the implementation and scientific claims.
This technical report is not the final six-page submission or presentation.

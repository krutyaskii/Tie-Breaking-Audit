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

## Validation and remaining experiment

Synthetic tests and validation pilots cover ranking formulae, exact ties,
filtering, candidate IDs, architecture, invalid inputs, bounds, canonical
order, seeded estimates and resume equivalence. The evidence bundle contains
actual CPU compatibility reports, pilot B/T and metrics, test logs and an
environment lock. These are preliminary validation results.

Local environment: Python 3.13.15, torch 2.11.0+cpu, NumPy 2.3.3, matplotlib
3.10.7 and psutil 7.2.2; no CUDA device. The ConvKB pilot takes roughly a
second per two queries, implying hours for a full CPU audit. The self-contained
Colab notebook and resumable runner provide the remaining GPU experiment.
No full test numbers or reproduction of the paper's effect size are claimed.
Acceptance requires 40932 valid queries per model and regenerated tables from
their shared B/T counts.

## Comparison and limitations

[Sun et al. (2020), Table 2](https://aclanthology.org/2020.acl-main.489.pdf)
reports FB15k-237 RotatE TOP/RANDOM/BOTTOM MRR 0.336/0.336/0.336, and ConvKB
0.407/0.243/0.130. Paper RANDOM averages five runs; our primary RANDOM is
analytic expectation, with five draws separately. Budget training, checkpoint
selection, PyTorch implementation and numerical environment differ. Matching
coordinate mixing does not prove full upstream recipe equivalence. A small
gap on these weights would be checkpoint-specific evidence, not proof that
the published result is false. Test results must not tune precision, seeds,
weights or training.

## Contribution and AI use

Our additions are shared filtered ranking, analytic metrics, seeded estimates,
per-query evidence, resumable streaming with source/environment guards,
independent architecture tests, standalone aggregation, tables, exact and
binned histograms and validation export. RotatE uses vendored kg-reeval code
with its recorded revision/license. ConvKB retains its existing public-model
attribution; model training was performed separately.

OpenAI Codex assisted with code, tests, documentation and local validation.
The team must review and explain the implementation and scientific claims.
This technical report is not the final six-page submission or presentation.

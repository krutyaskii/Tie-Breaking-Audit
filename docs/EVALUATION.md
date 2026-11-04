# Shared final test evaluation protocol

This document translates the team proposal into the evaluator handoff
contract. It specifies the final test audit for ConvKB and RotatE.
The completed RotatE package contains validation selection results and
zero scored test queries. This document supplies a protocol. The shared
evaluator's executable entry point remains to be implemented or integrated
by its owner.

## Split roles

| Split | Role | Triples |
| --- | --- | ---: |
| train | Fit model parameters and define training-negative exclusions | 272,115 |
| valid | Choose checkpoints and development settings | 17,535 |
| test | Compute the frozen final two-model audit | 20,466 |

All three splits contribute known true answers for filtered ranking.
That filtering use of test facts is part of this benchmark protocol.
Final test results must not drive checkpoint or hyperparameter selection.

Each test fact produces one head query `(?, r, t)` and one tail query
`(h, r, ?)`. The final query count is **40,932 per model**, with 14,541
candidate entities before filtering. The RotatE checkpoint is frozen at
update 30,000 with training seed 42. Record the corresponding ConvKB
checkpoint and its selection procedure before evaluating it.

## Set up the RotatE package

1. Extract the supplied handoff ZIP into a dedicated folder.
2. Verify `SHA256SUMS.json` using the snippet in
   [ROTATE_HANDOFF.md](ROTATE_HANDOFF.md#verify-the-bundle).
3. Use a suitable scoring environment and record the actual installed
   versions and device. `requirements-lock.txt` and `environment.json`
   record the original Colab versions. Reusing them provides the closest
   numerical match to the original run.
4. From the extracted folder, run `python prepare_data.py` to reconstruct
   and verify the dataset. This requires internet access on the first run.
5. Load `model.pt` through `RotatEAdapter.load` with the extracted
   dictionaries. The adapter rejects mismatched dictionary hashes.
6. Check the supplied validation integration examples before scoring test.
   The archived example scores were produced on CUDA. Record any device
   change when repeating the check.

The run used Python 3.13.16, torch 2.11.0+cu130, numpy 2.1.3, psutil 5.9.5,
CUDA 13.0, and a Tesla T4. CPU scoring is supported, with potentially different
float32 ties and runtime. Keep a single recorded device/environment for
each checkpoint's complete audit.

The verified model SHA-256 is
`5ec0bbc39e9568f6362ac38010cc752531dfbdba7aaea029661838707ce8b622`.
The verified archive SHA-256 is
`2c66eecac4fe499f8c3d695cb107d1c6025cda6758c875ea0461ba59421a0d1b`.

## Model interface

`RotatEAdapter.score_candidates(triples, mode)` accepts integer-ID triples
with shape `[batch, 3]` in head/relation/tail order and mode `head` or `tail`.
It returns a finite float32 NumPy array with shape `[batch, 14541]`.
Column `j` belongs to entity ID `j`. Larger values are better.

Read labels using the supplied dictionaries. Preserve their entity and
relation IDs across both model adapters. If ConvKB uses a different internal
indexing scheme, its adapter must explicitly map the shared input IDs and
candidate columns. Record and verify that mapping.

Use batches of eight queries or fewer for RotatE. Candidate chunking is
handled internally. Each query's raw scores must be computed once and
reused for every tie policy. A streaming implementation can retain the
per-query counts defined below instead of storing the full score matrix.
Record the scoring batch/chunk sizes with the environment.

## Filtering and exact ties

For tail prediction, exclude other entities `e` for which `(h, r, e)` occurs
in any of train, valid, or test. For head prediction, exclude other entities
`e` for which `(e, r, t)` occurs in any of the three splits. Always retain
the target entity. Filtering precedes rank and tie-count calculations.

Use the target's score from the same candidate-score row. A separately
computed single-triple score can differ slightly in float32 arithmetic.
Keep scores unrounded. Do not apply sigmoid, softmax, reduced precision,
or an approximate equality test when defining ties.

The core per-query calculation is independent of the model.

```python
target_score = scores[target_id]
keep = np.ones(len(scores), dtype=bool)
keep[list(other_known_true_ids)] = False
keep[target_id] = True
better = int(np.count_nonzero((scores > target_score) & keep))
tied = int(np.count_nonzero((scores == target_score) & keep))
assert tied >= 1
```

Here `tied` includes the target itself. A target tie occurs when `tied > 1`.
Ties among candidates at other score values do not change the target's rank.
The primary tie frequency is the fraction of queries with `tied > 1` after
filtering.

## Three ranking policies

Let `B` be `better` and `T` be `tied` for a query. Ranks are one-based.

| Quantity | Formula |
| --- | --- |
| Optimistic rank | `B + 1` |
| Pessimistic rank | `B + T` |
| Uniform-random rank | Uniform integer from `B + 1` through `B + T` |
| Optimistic reciprocal rank | `1 / (B + 1)` |
| Pessimistic reciprocal rank | `1 / (B + T)` |
| Expected random reciprocal rank | `sum(1 / (B + j) for j in 1..T) / T` |
| Expected random Hits@k | `clip(k - B, 0, T) / T` |

MRR and Hits@k average their query-level quantities over all 40,932 queries.
Report `k` equal to 1, 3, and 10 for each policy. The primary effect is
`MRR_optimistic - MRR_expected_random`, in absolute MRR units.

Expected random reciprocal rank averages reciprocals over possible ranks.
Taking the reciprocal of the mean rank yields a different quantity.
When `T = 1`, every policy gives the same rank `B + 1`.

For analytic random MRR, direct reciprocal summation or harmonic-number
differences can be used. Accumulate metrics in float64. Handle singleton
groups directly as `1 / (B + 1)` to avoid harmonic-subtraction cancellation.
The existing validation implementation provides this calculation in
`rotate.common.expected_metrics`.

## Five random evaluation seeds

The proposal specifies five random evaluation seeds. Use the explicit plan
`[0, 1, 2, 3, 4]` for the shared audit and save it in evaluation metadata.
This seed list is set before the test run. Keep query order deterministic
and document the RNG implementation and NumPy version.

Use test-file row order, all head queries first and all tail queries second.
For each seed, start a fresh `numpy.random.default_rng(seed)`. Given ordered
integer arrays `B` and `T`, sample one uniform position per target tie group.

```python
rng = np.random.default_rng(seed)
rank = B + rng.integers(low=1, high=T + 1)
mrr = np.mean(1.0 / rank)
hits = {k: np.mean(rank <= k) for k in (1, 3, 10)}
```

Save all five metric records, their arithmetic mean, and their sample
standard deviation using `ddof=1`. Compare the Monte Carlo mean to the
analytic expectation and retain the difference. Five draws need not equal
the expectation exactly. Every sampled metric must lie between the
pessimistic and optimistic endpoints.

These seeds change tie positions at fixed model weights. They do not create
five independently trained models. Keep training seed 42 separate in metadata.

## Hand-verifiable correctness examples

| Case | Retained scores, target marked with `*` | Expected result |
| --- | --- | --- |
| Unique target score | `[0.9, 0.8*, 0.7]` | `B=1`, `T=1`, rank 2, RR 1/2 for every policy |
| Target tie | `[0.9, 0.8*, 0.8, 0.1]` | `B=1`, `T=2`, ranks 2/3, expected RR 5/12 |
| All candidates tied | `[0.5*, 0.5, 0.5, 0.5]` | ranks 1/4, expected RR 25/48, expected Hits@1 1/4 |
| Filter another true answer | Before filtering `[0.9, 0.8*, 0.8, 0.1]`, exclude candidate 0 | `B=0`, `T=2`, ranks 1/2, expected RR 3/4 |

Also check a known true candidate tied with the target is removed when
appropriate, the target survives filtering, and both head/tail adapters
agree with their native score direction and candidate identities.

## Required evaluation outputs

The evaluator supplies the following artifacts for each model.

| Artifact | Required contents |
| --- | --- |
| Run metadata | Checkpoint hash and selected step, training seed, data/dictionary hashes, actual scoring environment/device, precision, batch/chunk sizes, filter rule, query order, random seeds, evaluator revision |
| Per-query record | Test row index, head/tail mode, target ID, `B`, and `T` |
| Main metric table | Optimistic, expected random, pessimistic MRR and Hits@1/3/10, optimistic-minus-expected gap, query counts |
| Tie statistics | Target-tie count/fraction and exact histogram of `T`, including singleton groups |
| Random-seed table | Five seeded MRR/Hits results and their means/sample standard deviations |
| Reproduction instructions | Pinned environment and one working command that generates the main results from frozen checkpoints |

An optional presentation histogram can group the number of *other* tied
candidates, `T - 1`, into bins `0`, `1-4`, `5-16`, `17-64`, `65-256`,
`257-1024`, `1025-4096`, and `4097-16384`. Preserve the exact histogram
alongside any binned figure. Count all queries, including those with no tie.

The per-query `B` and `T` arrays are sufficient to recompute the three
policies and five random-seed summaries without rerunning the model.
Keep both arrays associated with their checkpoint, split, and device hashes.

## Checks before accepting the final table

- Exactly 20,466 head and 20,466 tail queries are present for each model.
- Every target is retained and every `T` is at least one.
- Every query satisfies `B >= 0` and `B + T <= 14541`.
- Pessimistic MRR is at most expected random MRR, which is at most optimistic
  MRR, allowing only small float64 aggregation error.
- The tie histogram counts sum to 40,932 and the non-singleton count equals
  the reported target-tie count.
- Every policy and random seed uses the same frozen scores/counts.
- Both adapters pass mapping and score-direction checks.
- No validation metric appears in the final test-result table.

Compare each model's test MRR gap to
[Table 2](https://aclanthology.org/2020.acl-main.489.pdf). That table displays
RotatE MRR as 0.336 for all three policies and ConvKB as 0.407 TOP, 0.243
RANDOM, and 0.130 BOTTOM. Keep paper values and newly measured results
distinct, and explain configuration, checkpoint, precision, and protocol
differences. A negligible measured effect is an acceptable result.

## Responsibility boundary

The data/RotatE contribution supplies verified data reconstruction, frozen
weights, a scorer, logs, and documentation. The shared evaluator supplies
the complete final test run, random-seed cross-check, tie distributions,
ConvKB comparison, and final reproduction command. The final report is
updated from those artifacts after the audit is complete.

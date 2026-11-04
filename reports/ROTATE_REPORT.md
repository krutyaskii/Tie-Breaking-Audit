# Tie-Breaking Audit - Data and RotatE Experiment Report

Prepared on 2026-11-04. Scope covers the completed data preparation and RotatE
training contribution to course project E2. All measured ranking results in
this report come from validation. Final test evaluation is pending.

## Research question

Knowledge graph completion ranks possible entities for a missing head or tail
in a fact `(head, relation, tail)`. When several candidates receive the same
score, the target's position depends on how those ties are resolved. This
project asks how much that choice changes measured performance at fixed weights.

We use ConvKB and RotatE on FB15k-237. The primary outcome is optimistic MRR
minus analytically expected uniform-random MRR. Pessimistic ranking supplies
the opposite endpoint. We also measure target-tie frequency, tie-group sizes,
and Hits@1/3/10, following the team proposal and the protocol of
[Sun et al. (2020)](https://aclanthology.org/2020.acl-main.489/).

## Current experiment status

| Component | Evidence and status |
| --- | --- |
| Shared FB15k-237 preparation | Complete, original splits and IDs verified |
| RotatE training | Complete, 30,000 of 30,000 planned updates |
| Checkpoint selection | Complete, best full-validation MRR at update 30,000 |
| Scoring adapter verification | Passed on the exported checkpoint |
| Portable RotatE handoff | Complete, supplied archive verified locally |
| RotatE final test audit | Pending, handoff records zero test queries scored |
| ConvKB integration and test audit | Owned by the other model contributor and evaluator, status unverified here |
| Final two-model comparison | Pending shared test results |

The completed RotatE run is identified as `rotate_colab_budget`, with training
seed 42. It is a documented compute-limited configuration. The experimental
unit is this particular trained checkpoint.

## Dataset and preparation

We obtained FB15k-237 from the re-evaluation authors' repository at commit
`999b75e8a3ffa81973ea33bb22de17511b136bd1`. The archive SHA-256 is
`f8e19f3b4b24605c682e1982dcf37769d641ebac0e8a9f79d769867486ec752c`.
The [data manifest](data_manifest.json) records the source URL, split hashes,
dictionary hashes, sizes, and preparation checks.

| Item | Count |
| --- | ---: |
| Entities | 14,541 |
| Relations | 237 |
| Training triples | 272,115 |
| Validation triples | 17,535 |
| Test triples | 20,466 |

The original train, validation, test, and source dictionary files were preserved
byte-for-byte. The generated RotatE dictionaries reverse the dictionary
columns into `integer_id<TAB>label` while retaining every original ID. A small
debugging subset contains the first 100 validation rows.

Checks found no duplicate triples within a split and no exact triple overlap
between splits. Eight validation entities and 29 test entities are absent
from training. These entities and their original benchmark rows are retained.
The exact-overlap check does not establish absence of every possible leakage
mechanism.

Training positives and negative-sampling exclusions use training triples only.
Validation supplies checkpoint selection. The union of all three splits is
used to filter other known true answers when computing filtered ranks. Test
triples have supplied filtering facts, but no test ranking metrics were
computed during this training run.

## Model and implementation

RotatE represents entities in complex space and relations as rotations. The
score is a margin minus the distance between the rotated head embedding and
the tail embedding. Higher scores indicate more plausible facts.

We reuse the pinned public RotatE scorer and training objective distributed
with the re-evaluation study. Attribution, license, source hashes, and
compatibility changes are recorded in `vendor/kg_reeval/SOURCE.json`.
Compatibility edits address imports and NumPy's removed `in1d` function.
Checks confirm that the upstream `RotatE` and `train_step` method bodies
retain their recorded syntax-tree hashes.

The runner uses self-adversarial negative sampling, subsampling weights,
alternating head/tail batches, Adam, and the upstream learning-rate decay
convention. Microbatches accumulate gradients with weights normalized over
the effective batch. The runner saves model, optimizer, sampler, and random
states for resumption. Model selection uses analytically expected random
filtered MRR over the complete validation split every 3,000 updates.

Training and scoring use float32 with mixed precision and TF32 disabled.
The raw-score adapter preserves candidate-ID order and performs no ranking
or filtering. The shared evaluator owns those operations. The legacy upstream
`test_step` is bypassed in favor of the project's checked ranking procedure.

## Configuration and computing environment

| Setting | Completed run |
| --- | ---: |
| Complex embedding dimension | 200 |
| Entity embedding real-valued components | 400 |
| Gamma | 12 |
| Effective batch size | 512 |
| Microbatch size | 64 |
| Negative samples per positive | 128 |
| Initial learning rate | 0.0001 |
| Adversarial temperature | 1 |
| Planned and completed optimizer updates | 30,000 |
| Learning-rate warm-up threshold | 15,000 |
| Training seed | 42 |
| Validation interval | 3,000 updates |
| Validation size | 17,535 triples, 35,070 queries |
| Query batch size | 8 |
| Candidate chunk size | 2,048 |

Training ran on a Google Colab Tesla T4 with Python 3.13.16, PyTorch
2.11.0+cu130, NumPy 2.1.3, psutil 5.9.5, and CUDA 13.0. The recorded runner
elapsed time is 4,755.176 seconds, approximately 79 minutes 15 seconds. This
includes its validation and checkpoint work and excludes data preparation
and final handoff export.

Observed peak process RSS was 1,642,106,880 bytes. PyTorch peak allocated GPU
memory was 187,908,096 bytes and peak reserved memory was 228,589,568 bytes.
These are process/PyTorch measurements, rather than total Colab GPU usage.

## Measured validation results

For each validation fact, both head and tail prediction rank all 14,541
entities before filtering. Other known true answers are excluded and the
target is retained. Ties use exact equality of unrounded adapter float32
scores. The selected checkpoint is update 30,000.

| Metric | Selected checkpoint, validation |
| --- | ---: |
| Optimistic MRR | 0.2863605780470804 |
| Expected random MRR | 0.2863605777207394 |
| Pessimistic MRR | 0.2863605773943984 |
| Optimistic minus expected random MRR | 3.2634101e-10 |
| Expected random Hits@1 | 0.2029654975762760 |
| Expected random Hits@3 | 0.3164813230681494 |
| Expected random Hits@10 | 0.4540918163672655 |
| Queries with another candidate tied with the target | 24 / 35,070 |
| Target-tie fraction | 0.0006843455945252353 |
| Target-tie percentage | 0.06843456% |

The count of 24 is recovered from the recorded fraction and query count.
Validation logs contain aggregate metrics. They do not preserve per-query
tie counts or a tie-group size distribution.

| Optimizer update | Validation expected random MRR |
| --- | ---: |
| 0 | 0.0010393311 |
| 3,000 | 0.0135786348 |
| 6,000 | 0.0602170273 |
| 9,000 | 0.1686319793 |
| 12,000 | 0.2383538970 |
| 15,000 | 0.2739592279 |
| 18,000 | 0.2777026680 |
| 21,000 | 0.2801701534 |
| 24,000 | 0.2823169041 |
| 27,000 | 0.2846215962 |
| 30,000 | 0.2863605777 |

The improvement over initialization shows that the model learned useful
ranking structure. At the selected checkpoint, about 20.3% of validation
targets rank first and about 45.4% fall in the top ten under expected random
ranking. These percentages describe filtered candidate-ranking queries.

The preliminary audit observation is that tie policy has a negligible effect
on aggregate validation MRR for this checkpoint. Ties remain present in
24 queries, so identical displayed MRR values would not imply zero ties.
Training completion means the planned update budget was reached. Validation
MRR was still increasing at the final recorded evaluation.

## Relation to the paper and limitations

[Table 2 of Sun et al.](https://aclanthology.org/2020.acl-main.489.pdf)
reports RotatE MRR of 0.336 under RANDOM, TOP, and BOTTOM at the table's
displayed precision. ConvKB changes from 0.243 under RANDOM to 0.407 under
TOP, an absolute gap of 0.164. These are literature reference values on
FB15k-237 test data.

Our validation observation is qualitatively consistent with RotatE's low
sensitivity in that table. A numerical test comparison awaits the shared
evaluator. The current validation MRR cannot fill a test-result row.

The completed configuration differs from the provided paper preset in
embedding dimension, margin, effective batch size, negative sample count,
and training budget. It uses 200 complex dimensions and 30,000 updates,
whereas the provided larger preset uses 1,000 dimensions and 150,000 updates.
The weights are newly trained. Absolute reproduction of the paper's MRR has
not been established.

There is one training seed. The five planned evaluation seeds measure
tie-breaking variability at fixed weights and cannot estimate training
variability. Exact float32 ties can also depend on device and numerical
environment. Each model must be scored once on a recorded device and those
same scores must drive every tie policy.

## Verification and handoff provenance

The supplied archive is
`rotate_colab_budget_handoff_20261102_130712.zip`. Its SHA-256 is
`2c66eecac4fe499f8c3d695cb107d1c6025cda6758c875ea0461ba59421a0d1b`.
All 33 entries listed in its `SHA256SUMS.json` matched during inspection.
The exported model SHA-256 is
`5ec0bbc39e9568f6362ac38010cc752531dfbdba7aaea029661838707ce8b622`.

The archive's adapter checks passed on `cuda:0`, covering eight validation
triples in both directions, all entity candidates, exact repeatability,
dictionary hashes, and native-scorer agreement. The maximum absolute
difference from native single-triple scoring was 1.9073486328125e-6 for head
prediction and zero for tail prediction. That numerical check uses a
tolerance. The tie definition uses exact adapter-score equality.

The implementation also has seven unit checks covering manually verifiable
scores, candidate-ID alignment, the microbatch update against upstream,
training-negative exclusions, sampler/random-state resumption, filtering
and expected reciprocal rank, and checkpoint/mapping compatibility. These
checks were run during pipeline preparation. The present documentation
update reverified archive checksums and the consistency of reported metadata.

The bundle contains weights, dictionaries, source and license, configuration,
environment, logs, data provenance, and validation integration examples.
Dataset splits are reconstructed separately. Resume optimizer state stays
in the training run directory and is excluded from the evaluator package.
See the [handoff guide](../docs/ROTATE_HANDOFF.md) for loading instructions.
The small [run evidence record](rotate_colab_budget_run.json) preserves
the inspected metadata and full validation learning curve.

## Remaining work and acceptance criteria

| Owner | Remaining work | Done when |
| --- | --- | --- |
| Data and RotatE owner | Supply the frozen archive and documentation | Evaluator can verify hashes, reconstruct data, and load scores with matching IDs |
| ConvKB owner | Supply a verified checkpoint and adapter | Scores follow the shared entity order and documented direction, and checkpoint selection is recorded |
| Shared evaluator | Audit both checkpoints on the complete test split | Each model has 40,932 queries, three policies, MRR and Hits@1/3/10, target-tie frequency, and tie-size distributions |
| Shared evaluator | Run five random evaluation seeds | Seeds, per-seed results, means, and sample standard deviations are saved alongside the analytic expectation |
| Shared evaluator | Deliver reproducible outputs | Per-query counts, data/model hashes, device, pinned environment, and a working evaluation command accompany the main table |
| Team | Finish the project comparison | Test gaps are compared with Table 2, discrepancies are explained, and the final report includes both models |

The detailed [evaluation protocol](../docs/EVALUATION.md) specifies the
ranking formulas, seed plan, hand-verifiable examples, and output contract.
The shared evaluator's final one-command entry point remains to be provided.

## Our contribution and AI use

Our contribution in the completed data and RotatE work consists of verified
dataset reconstruction, an ID-preserving interface, a resumable training
runner with microbatch accumulation, validation-based model selection, a
raw-score adapter, correctness checks, a self-contained Colab workflow, and
a portable handoff with provenance. The scorer and learning objective are
credited to the public upstream implementation. The shared two-model test
audit remains a team deliverable.

OpenAI Codex assisted with code preparation, compatibility work, verification,
and documentation drafting. The Colab run was executed by the project member.
The team is responsible for reviewing and understanding the implementation,
evaluation choices, and reported claims before submission.

## Submission scope and references

This report supplies the completed data/RotatE section and preliminary
evidence for the midway checkpoint. The
[course requirements](https://lukmanovr.github.io/dkr/project/index.html)
specify a final six-page report in NeurIPS style with reproducible code,
an explicit contribution paragraph, and an AI-use statement. The final team
report needs the shared test results and formatting step.

- Sun et al. (2020), [A Re-evaluation of Knowledge Graph Completion Methods](https://aclanthology.org/2020.acl-main.489/).
- [Pinned upstream implementation and data](https://github.com/svjan5/kg-reeval/tree/999b75e8a3ffa81973ea33bb22de17511b136bd1).
- [Course topic E2](https://lukmanovr.github.io/dkr/project/topics.html#track-e-reproducibility-studies).
- Team proposal, *E2 - The tie-breaking audit*, Maria Karpova, Georgij Pyanov, and Daria Potapova.

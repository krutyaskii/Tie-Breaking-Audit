# Tie-Breaking Audit

Course project E2: reproduce the evaluation issue described in
[Sun et al., ACL 2020](https://aclanthology.org/2020.acl-main.489/).
We will evaluate ConvKB and RotatE on FB15k-237 and measure how optimistic,
pessimistic, and uniform random tie-breaking change ranking metrics at fixed
model weights.

The project includes shared data preparation and a RotatE training/scoring
pipeline. The completed `rotate_colab_budget` run trained for 30,000 updates
on a Colab Tesla T4. Its selected checkpoint is ready for the shared evaluator.
The supplied handoff records **zero scored test queries**. The reported MRR
of 0.286361 belongs to validation checkpoint selection.

## Completed run and evaluation status

`audit.py` is a work-in-progress streaming evaluator, with reproducible
aggregation from per-query counts. The existing ConvKB checkpoint is a legacy
architecture variant and cannot establish a ConvKB replication; its
correction and retraining are pending. No full test audit is yet reported.
See [current audit instructions](docs/AUDIT_STATUS.md).

- [RotatE experiment report](reports/ROTATE_REPORT.md) covers the data,
  implementation, completed run, measured validation results, and limitations.
- [Verified run evidence](reports/rotate_colab_budget_run.json) preserves the
  supplied archive's metadata and validation history with integrity checks.
- [Shared test evaluation protocol](docs/EVALUATION.md) defines the final
  audit on 20,466 test triples, or 40,932 head/tail queries, and the remaining
  deliverables for both model owners and the evaluator.
- [Evaluator handoff guide](docs/ROTATE_HANDOFF.md) explains how to load the
  frozen checkpoint and score candidates with the original IDs.

Final test metrics, tie-group distributions, five random evaluation seeds,
and the ConvKB comparison remain to be produced by the shared evaluator.
The report documents the completed data and RotatE contribution. The final
team report will incorporate those test results.

## Obtain the trained checkpoint

The trained evaluator package is shared separately by the RotatE owner.
Request `rotate_colab_budget_handoff_20261102_130712.zip` (22,774,609 bytes).
It includes the selected weights, matching dictionaries, scoring code,
configuration, environment, and training/validation logs. Dataset splits
are reconstructed using the included preparation script.

Verify the archive against the committed
[SHA-256 sidecar](reports/rotate_colab_budget_handoff_20261102_130712.zip.sha256)
before extraction. Then follow the
[handoff instructions](docs/ROTATE_HANDOFF.md). The
[run evidence](reports/rotate_colab_budget_run.json) also records the archive
and model hashes. The checkpoint package is distributed separately from Git.

## Train RotatE in Colab

Upload [RotatE_Colab.ipynb](notebooks/RotatE_Colab.ipynb) to
[Google Colab](https://colab.research.google.com/), choose a T4 GPU, and run
the cells in order. The notebook embeds the code and needs no GitHub push.
It can save resumable checkpoints to your Google Drive and export a package
for the shared evaluator.

Read [the training guide](docs/ROTATE.md) before selecting a preset.
The default `colab_budget` run differs from the paper's hyperparameters; the
larger `paper` preset is also provided. Weights are selected using validation
only. No final test metrics are computed by the training pipeline.

The evaluator interface is documented in [the handoff guide](docs/ROTATE_HANDOFF.md).

## Quick start

Requirements: Git, Python 3.10 or newer, and an internet connection for the
first download. The data preparation script has no third-party dependencies.

```sh
git clone https://github.com/krutyaskii/Tie-Breaking-Audit.git
cd Tie-Breaking-Audit
python prepare_data.py
```

Use `python3` instead of `python` if that is your Python command on Linux/macOS.
On Windows, `py -3 prepare_data.py` is another option.

The script downloads the source archive from a fixed upstream commit, verifies
its SHA-256 checksum, preserves the original splits and IDs, prepares the
RotatE dictionaries, and checks all output files against the committed manifest.
Repeated runs reuse the local archive and verify the same results.

For file formats, instructions for each team role, manual downloads, and
troubleshooting, see **[the full data guide](docs/DATA.md)**.

## Expected dataset

| Item | Count |
| --- | ---: |
| Entities | 14,541 |
| Relations | 237 |
| Training triples | 272,115 |
| Validation triples | 17,535 |
| Test triples | 20,466 |

The verified source has no duplicate rows within a split and no exact triple
overlap between splits. Validation contains 8 entities absent from training;
test contains 29. These rows are preserved. See
[the manifest](reports/data_manifest.json) for the full audit and file hashes.

## Repository layout

```text
prepare_data.py              # Download, prepare, and verify the shared dataset
docs/DATA.md                # Full setup and usage instructions
reports/data_manifest.json  # Versioned provenance, statistics, and SHA-256 hashes
raw/FB15k-237.zip            # Downloaded locally; ignored by Git
data/FB15k-237/              # Generated locally; ignored by Git
```

## Data policy

Dataset files, archives, checkpoints, and training runs stay out of Git.
Commit the preparation code, instructions, and small provenance reports instead.
Each teammate runs the same command to get the same data.

Train on `train.txt`, select settings/checkpoints using `valid.txt`, and use
`test.txt` for final evaluation. Both models use the same underlying triples
and entity/relation identities. Checkpoint compatibility must be checked before
using externally trained weights.

## References

- Sun et al. (2020), [A Re-evaluation of Knowledge Graph Completion Methods](https://aclanthology.org/2020.acl-main.489/).
- [Upstream code and data archive](https://github.com/svjan5/kg-reeval/tree/999b75e8a3ffa81973ea33bb22de17511b136bd1).
- [Upstream RotatE data loader](https://github.com/svjan5/kg-reeval/blob/999b75e8a3ffa81973ea33bb22de17511b136bd1/RotatE/codes/run.py).

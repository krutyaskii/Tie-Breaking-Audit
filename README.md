# Tie-Breaking Audit

Course project E2: reproduce the evaluation issue described in
[Sun et al., ACL 2020](https://aclanthology.org/2020.acl-main.489/).
We will evaluate ConvKB and RotatE on FB15k-237 and measure how optimistic,
pessimistic, and uniform random tie-breaking change ranking metrics at fixed
model weights.

The current repository contains the shared data setup. Model training,
checkpoint adapters, and the shared evaluator are the next stages.

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

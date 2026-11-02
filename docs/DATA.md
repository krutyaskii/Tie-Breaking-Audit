# FB15k-237: shared data setup

This guide is for all three project roles: RotatE, ConvKB, and the shared
evaluator. Everyone uses the same source triples and mappings.

## 1. Download and prepare

Install Git and Python 3.10 or newer. No Python packages are needed for this step.

From a terminal:

```sh
git clone https://github.com/krutyaskii/Tie-Breaking-Audit.git
cd Tie-Breaking-Audit
python prepare_data.py
```

If you already have the repository, open its root folder and run only the last
command. Use `python3 prepare_data.py` on systems where Python is named
`python3`, or `py -3 prepare_data.py` on Windows.

The first run downloads an approximately 8.8 MB ZIP archive. Allow about 40 MB
of disk space for the archive and prepared files. Subsequent runs use the cached
archive, so they do not need an internet connection.

The final output must include `READY`, the expected counts below, and the path
to the manifest. A failed validation exits with a nonzero status. The script
works regardless of the current terminal directory because output paths are
resolved relative to `prepare_data.py`.

This command prepares the data only. It does not install model libraries,
download trained checkpoints, train a model, or compute evaluation metrics.

## 2. Files you will get

All prepared files are in `data/FB15k-237/`:

| File | Purpose | Rows |
| --- | --- | ---: |
| `train.txt` | Model training | 272,115 |
| `valid.txt` | Choosing settings and checkpoints | 17,535 |
| `test.txt` | Final evaluation | 20,466 |
| `entities.dict` | Entity mapping in RotatE format | 14,541 |
| `relations.dict` | Relation mapping in RotatE format | 237 |
| `entity2id.txt` | Original entity mapping | 14,541 |
| `relation2id.txt` | Original relation mapping | 237 |
| `debug_valid_100.txt` | First 100 validation triples for debugging | 100 |

### Triple format

Each line contains three tab-separated fields in this order:

```text
head_label<TAB>relation_label<TAB>tail_label
```

For example, the first training triple is:

```text
/m/027rn<TAB>/location/country/form_of_government<TAB>/m/06cx9
```

`<TAB>` represents an actual tab character in the files. Names such as
`/m/027rn` are source identifiers; do not rename or translate them.

### Mapping formats

The original `entity2id.txt` and `relation2id.txt` use:

```text
label<TAB>integer_id
```

The added `entities.dict` and `relations.dict` use the format expected by RotatE:

```text
integer_id<TAB>label
```

For example, `/m/027rn` has ID `0` in both entity dictionaries. Preparation
reverses the column order for the RotatE files and sorts them by ID; it does
not assign new IDs. Entity IDs range from 0 to 14,540 and relation IDs from
0 to 236.

## 3. What preparation changes

The script performs these steps:

1. Downloads the archive from a fixed commit in the paper's repository.
2. Checks the archive's SHA-256 against the recorded value.
3. Reads the three splits and two source dictionaries directly from the ZIP.
4. Validates triple formats, dictionary IDs, row counts, and split overlaps.
5. Copies the original five files byte for byte, including their line endings.
6. Creates the two RotatE dictionaries without changing source IDs.
7. Creates a 100-row validation subset for debugging.
8. Verifies the generated manifest against the committed reference.

The original dataset split is preserved: there is no reshuffling, resplitting,
deduplication, or removal of entities. The source archive also includes
`*.init` embedding files. They remain in the cached ZIP but are not copied into
the prepared data folder. They are not a trained RotatE checkpoint.

## 4. Verified properties and limitations

The recorded data checks show:

- Every triple has exactly three fields.
- Every entity and relation label is present in its dictionary.
- Mapping IDs are unique and contiguous from zero.
- Split sizes match the table above.
- Duplicate rows within each split: 0.
- Exact triple intersections for train/valid, train/test, and valid/test: 0.
- Entities absent from train: 8 in validation and 29 in test.
- Relations absent from train: 0 in both validation and test.

The unseen entities are part of the source splits and are retained. Record
this limitation when interpreting results. The overlap check detects exact
triple overlap; it does not prove that all possible forms of leakage are absent.

## 5. Instructions for the RotatE owner

Use `data/FB15k-237/` as the dataset directory. The upstream RotatE loader reads
these five files:

```text
train.txt
valid.txt
test.txt
entities.dict
relations.dict
```

Set its `--data_path` argument to the absolute or relative path of that folder.
Relative paths are resolved by the model from its current working directory.
Do not confuse FB15k-237 with FB15k: verify the counts printed when the model
loads the data.

Model dependencies and a complete training command will be documented with
the RotatE implementation. Start with a short loading/scoring check before
spending time on a full training run.

Before using an external checkpoint, verify that every entity and relation
label has the same ID as in the checkpoint's training data. A matching dataset
name or dictionary size is insufficient. If its IDs differ, implement and
verify an explicit conversion of the model inputs/embeddings; do not silently
replace the team's dictionaries.

## 6. Instructions for the ConvKB owner

Use the same `train.txt`, `valid.txt`, and `test.txt`. For implementations from
the re-evaluation repository, the original mappings are available as
`entity2id.txt` and `relation2id.txt`.

Check the selected implementation's loader before training. If it requires a
different file format, adapt the input format while preserving the triple
identities and their dictionary correspondence. Document any conversion.

Check external checkpoint mappings as described for RotatE. The shared data
setup does not by itself verify that a checkpoint is compatible.

## 7. Instructions for the evaluator owner

- Read all three splits and the mappings from this same data directory.
- Use the validation split for development and the test split for final results.
- Evaluate both head prediction `(?, relation, tail)` and tail prediction
  `(head, relation, ?)` against all 14,541 entity candidates.
- For filtered evaluation, use the union of train, valid, and test to exclude
  other known true answers. Keep the target triple being evaluated.
- Using held-out facts for this standard filtering step is separate from
  training: they must not be used as positive training examples or for choosing
  model weights based on test results.
- Compute each query's scores once per checkpoint and reuse them for optimistic,
  pessimistic, and uniform random tie-breaking.
- Define ties using exact equality of unrounded float32 scores, as specified
  in the proposal.
- `debug_valid_100.txt` is only a debugging subset. Clearly label any results
  computed on it; it is not a replacement for full validation or test evaluation.

The evaluator itself is not included in this data setup.

## 8. Provenance and reproducibility

The committed `reports/data_manifest.json` contains the source URL and commit,
archive checksum, per-file SHA-256 checksums and byte sizes, split statistics,
and preparation rules. It contains metadata, not dataset rows.

Source repository:
[svjan5/kg-reeval](https://github.com/svjan5/kg-reeval).

Fixed source commit:
`999b75e8a3ffa81973ea33bb22de17511b136bd1`.

Expected archive SHA-256:

```text
f8e19f3b4b24605c682e1982dcf37769d641ebac0e8a9f79d769867486ec752c
```

Source article:
[A Re-evaluation of Knowledge Graph Completion Methods](https://aclanthology.org/2020.acl-main.489/),
Sun et al., ACL 2020.

The team should share this repository and run the preparation command rather
than exchange edited copies of the dataset. A successful run checks against
the same manifest, so everyone can confirm identical prepared files.

## 9. Troubleshooting

### Python is not found

Try `python3` on Linux/macOS or `py -3` on Windows. Install Python 3.10 or newer
if neither is available.

### The automatic download fails

Check your internet connection. You can also download the archive in a browser:

[Download the pinned FB15k-237 archive](https://raw.githubusercontent.com/svjan5/kg-reeval/999b75e8a3ffa81973ea33bb22de17511b136bd1/data_compressed/FB15k-237.zip).

Create `raw/` in the repository root, save the file as `raw/FB15k-237.zip`,
and run `python prepare_data.py` again. Do not extract it manually: the script
reads and verifies the ZIP itself.

### The archive checksum does not match

Do not train on that copy. Move the suspect archive outside `raw/` for
inspection, then download again from the pinned link. Keep the checksum check
enabled. If a fresh download still differs, report it as a blocker.

### An existing prepared file or manifest differs

The script stops instead of overwriting it. Inspect or back up the changed
file first. To rebuild a modified prepared dataset, move the generated
`data/FB15k-237/` folder outside `data/`, then rerun the script. If the reference
manifest was edited, restore the committed version after reviewing the change.
Do not change reference hashes simply to make a failing check pass.

### Which files should be committed?

Commit code, English documentation, and the small reference manifest.
The `.gitignore` excludes `raw/`, `data/`, checkpoints, and training runs.
Do not add generated files with `git add -f`.

### Why are the splits not committed?

The archive is small enough to download quickly, but dataset copies would add
unnecessary Git history. A pinned source, checksum verification, and one setup
command reproduce the data without storing it in this repository.

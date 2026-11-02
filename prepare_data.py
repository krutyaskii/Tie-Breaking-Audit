"""Prepare the paper's FB15k-237 files for RotatE, using Python's stdlib only."""

import hashlib
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
COMMIT = "999b75e8a3ffa81973ea33bb22de17511b136bd1"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/svjan5/kg-reeval/{COMMIT}"
    "/data_compressed/FB15k-237.zip"
)
ARCHIVE_SHA256 = "f8e19f3b4b24605c682e1982dcf37769d641ebac0e8a9f79d769867486ec752c"
ARCHIVE = ROOT / "raw" / "FB15k-237.zip"
DEST = ROOT / "data" / "FB15k-237"
SPLIT_COUNTS = {"train": 272115, "valid": 17535, "test": 20466}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def save_unchanged_or_new(path, content):
    """Do not silently replace a file that a teammate may have modified."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing file differs; inspect it before replacing: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def read_mapping(content, expected_count):
    mapping = {}
    for line in content.decode("utf-8").splitlines():
        label, number = line.split("\t")
        if label in mapping:
            raise ValueError(f"Duplicate mapping label: {label}")
        mapping[label] = int(number)
    if len(mapping) != expected_count:
        raise ValueError(f"Expected {expected_count} dictionary entries, got {len(mapping)}")
    if set(mapping.values()) != set(range(expected_count)):
        raise ValueError("Dictionary IDs must be unique and contiguous from zero")
    return mapping


def main():
    if not ARCHIVE.exists():
        with urlopen(SOURCE_URL, timeout=60) as response:
            downloaded = response.read()
        if sha256(downloaded) != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive SHA-256 does not match the recorded source")
        save_unchanged_or_new(ARCHIVE, downloaded)

    archive_bytes = ARCHIVE.read_bytes()
    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch; stop before using these data")

    names = ["train.txt", "valid.txt", "test.txt", "entity2id.txt", "relation2id.txt"]
    with ZipFile(ARCHIVE) as archive:
        files = {name: archive.read(f"FB15k-237/{name}") for name in names}

    entities = read_mapping(files["entity2id.txt"], 14541)
    relations = read_mapping(files["relation2id.txt"], 237)
    # RotatE expects ID<TAB>label. Preserve the source IDs; only reverse columns.
    for filename, mapping in [("entities.dict", entities), ("relations.dict", relations)]:
        files[filename] = "".join(
            f"{number}\t{label}\n"
            for label, number in sorted(mapping.items(), key=lambda item: item[1])
        ).encode("utf-8")

    split_sets = {}
    split_entities = {}
    split_relations = {}
    split_stats = {}
    for split, expected in SPLIT_COUNTS.items():
        triples = []
        for line_number, line in enumerate(files[f"{split}.txt"].decode("utf-8").splitlines(), 1):
            fields = line.split("\t")
            if len(fields) != 3:
                raise ValueError(f"{split}:{line_number}: expected head, relation, tail")
            head, relation, tail = fields
            if head not in entities or tail not in entities or relation not in relations:
                raise ValueError(f"{split}:{line_number}: unknown entity or relation")
            triples.append((head, relation, tail))
        if len(triples) != expected:
            raise ValueError(f"{split}: expected {expected} triples, got {len(triples)}")
        split_sets[split] = set(triples)
        split_entities[split] = {entity for head, _, tail in triples for entity in (head, tail)}
        split_relations[split] = {relation for _, relation, _ in triples}
        split_stats[split] = {
            "triples": len(triples),
            "unique_triples": len(split_sets[split]),
            "duplicate_rows": len(triples) - len(split_sets[split]),
            "entities_present": len(split_entities[split]),
            "relations_present": len(split_relations[split]),
        }

    overlaps = {
        f"{left}_{right}": len(split_sets[left] & split_sets[right])
        for left, right in [("train", "valid"), ("train", "test"), ("valid", "test")]
    }
    if any(overlaps.values()):
        raise ValueError(f"Exact triples overlap between splits: {overlaps}")

    coverage = {
        split: {
            "entities_absent_from_train": len(split_entities[split] - split_entities["train"]),
            "relations_absent_from_train": len(split_relations[split] - split_relations["train"]),
        }
        for split in ("valid", "test")
    }
    # Small debugging set, drawn only from validation; never a replacement for test.
    validation_lines = files["valid.txt"].decode("utf-8").splitlines()
    files["debug_valid_100.txt"] = ("\n".join(validation_lines[:100]) + "\n").encode("utf-8")
    manifest = {
        "dataset": "FB15k-237",
        "source_repository": "https://github.com/svjan5/kg-reeval",
        "source_commit": COMMIT,
        "source_url": SOURCE_URL,
        "source_archive_sha256": ARCHIVE_SHA256,
        "entities": len(entities),
        "relations": len(relations),
        "splits": split_stats,
        "exact_triple_overlap": overlaps,
        "train_coverage": coverage,
        "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in files.items()},
        "preparation": {
            "split_bytes_changed": False,
            "source_ids_preserved": True,
            "rotate_dictionary_format": "integer_id<TAB>label",
            "triples_format": "head_label<TAB>relation_label<TAB>tail_label",
            "debug_subset": "First 100 validation rows, for debugging only",
            "checkpoint_warning": "External checkpoints must use exactly these entity and relation IDs",
            "filtering": "Use train + valid + test to filter other true answers; retain target",
            "training": "Train on train only; select checkpoints using validation only",
            "overlap_check_scope": "Exact duplicate triples only; not a full audit of all leakage types",
        },
    }
    # All validation is complete before publishing prepared files.
    for filename, data in files.items():
        existing = DEST / filename
        if existing.exists() and existing.read_bytes() != data:
            raise ValueError(f"Prepared file was changed; inspect it first: {existing}")
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    existing_manifest = ROOT / "reports" / "data_manifest.json"
    if existing_manifest.exists() and existing_manifest.read_bytes() != manifest_bytes:
        raise ValueError(f"Manifest differs; inspect it first: {existing_manifest}")
    for filename, data in files.items():
        save_unchanged_or_new(DEST / filename, data)
    save_unchanged_or_new(existing_manifest, manifest_bytes)
    print(f"READY: {DEST}")
    print(f"Entities: {len(entities)}; relations: {len(relations)}")
    for split, stats in split_stats.items():
        print(f"{split}: {stats['triples']} triples, {stats['duplicate_rows']} duplicate rows")
    print(f"Exact triple overlaps: {overlaps}")
    print(f"Validation/test coverage relative to train: {coverage}")
    print(f"Manifest: {existing_manifest}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, URLError, ValueError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        print(
            f"If the download failed, download {SOURCE_URL}\n"
            f"in your browser, save it as {ARCHIVE}, and run this script again.\n"
            "If an existing file differs, inspect the change before replacing it.\n"
            "See docs/DATA.md for setup and troubleshooting.",
            file=sys.stderr,
        )
        sys.exit(1)

"""Shared data verification, provenance, and validation ranking."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def environment():
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "psutil")}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def source_provenance():
    paths = [*ROOT.glob("rotate/*.py"), *ROOT.glob("vendor/kg_reeval/*.py"),
             ROOT / "vendor/kg_reeval/SOURCE.json"]
    result = {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in sorted(paths)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return {"base_git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "file_sha256": result}


def read_dictionary(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        number, label = line.split("\t")
        result[label] = int(number)
    if set(result.values()) != set(range(len(result))):
        raise ValueError(f"Invalid dictionary IDs: {path}")
    return result


def load_data(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((ROOT / "reports/data_manifest.json").read_text(encoding="utf-8"))
    for name, reference in manifest["files"].items():
        if file_hash(data_dir / name) != reference["sha256"]:
            raise ValueError(f"Dataset differs from the reference: {name}; run prepare_data.py")
    entities = read_dictionary(data_dir / "entities.dict")
    relations = read_dictionary(data_dir / "relations.dict")
    splits = {}
    for split in ("train", "valid", "test"):
        rows = []
        for line in (data_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            h, r, t = line.split("\t")
            rows.append((entities[h], relations[r], entities[t]))
        splits[split] = np.asarray(rows, dtype=np.int64)
    return splits, manifest


def truth_indexes(splits):
    heads, tails = defaultdict(set), defaultdict(set)
    for rows in splits.values():
        for h, r, t in rows.tolist():
            heads[(r, t)].add(h)
            tails[(h, r)].add(t)
    return heads, tails


def expected_metrics(better, tied):
    better, tied = np.asarray(better, dtype=np.int64), np.asarray(tied, dtype=np.int64)
    if np.any(tied < 1):
        raise ValueError("The target must be included in its own tie group")
    harmonic = np.r_[0.0, np.cumsum(1.0 / np.arange(1, int((better + tied).max()) + 1))]
    expected_rr = (harmonic[better + tied] - harmonic[better]) / tied
    # A singleton tie group has an exact reciprocal rank; avoid cancellation.
    expected_rr[tied == 1] = 1.0 / (better[tied == 1] + 1)
    result = {
        "MRR_expected_random": float(np.mean(expected_rr)),
        "MRR_optimistic": float(np.mean(1.0 / (better + 1))),
        "MRR_pessimistic": float(np.mean(1.0 / (better + tied))),
        "target_tie_fraction": float(np.mean(tied > 1)),
        "queries": int(len(tied)),
    }
    for k in (1, 3, 10):
        result[f"Hits@{k}_expected_random"] = float(np.mean(np.clip(k - better, 0, tied) / tied))
    return result


def validation_metrics(adapter, triples, indexes, batch_size):
    better_counts, tie_counts = [], []
    for mode in ("head", "tail"):
        for start in range(0, len(triples), batch_size):
            batch = triples[start:start + batch_size]
            scores = adapter.score_candidates(batch, mode)
            for row, (h, r, t) in enumerate(batch.tolist()):
                target = h if mode == "head" else t
                target_score = scores[row, target]
                known = indexes[0][(r, t)] if mode == "head" else indexes[1][(h, r)]
                keep = np.ones(adapter.nentity, dtype=bool)
                keep[list(known)] = False
                keep[target] = True
                better_counts.append(int(np.count_nonzero((scores[row] > target_score) & keep)))
                tie_counts.append(int(np.count_nonzero((scores[row] == target_score) & keep)))
    return expected_metrics(better_counts, tie_counts)

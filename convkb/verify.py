"""Sanity checks for a ConvKB checkpoint before handoff."""

import argparse
import json
from pathlib import Path

import numpy as np

from convkb.adapter import ConvKBAdapter
from convkb.common import ROOT, file_hash, load_data, write_json


def verify(checkpoint, data_dir, device, examples_out):
    adapter = ConvKBAdapter.load(checkpoint, data_dir, device=device)
    splits, manifest = load_data(data_dir)
    triples = splits["valid"][:8]
    head_scores = adapter.score_candidates(triples, mode="head")
    tail_scores = adapter.score_candidates(triples, mode="tail")
    if head_scores.shape != (8, adapter.nentity) or tail_scores.shape != (8, adapter.nentity):
        raise ValueError("Unexpected score shape")
    if head_scores.dtype != np.float32 or tail_scores.dtype != np.float32:
        raise ValueError('Expected float32 scores')
    if not np.all(np.isfinite(head_scores)) or not np.all(np.isfinite(tail_scores)):
        raise ValueError("Non-finite scores")
    repeat_head = bool(np.array_equal(head_scores, adapter.score_candidates(triples, mode="head")))
    repeat_tail = bool(np.array_equal(tail_scores, adapter.score_candidates(triples, mode="tail")))
    if not repeat_head or not repeat_tail:
        raise ValueError("Scores are not exactly repeatable")
    np.savez(examples_out, triples=triples, head_scores=head_scores, tail_scores=tail_scores)
    report = {
        "passed": True,
        "checkpoint_sha256": file_hash(checkpoint),
        "device": device,
        "validation_triples_checked": int(len(triples)),
        "entity_candidates_per_query": int(adapter.nentity),
        "relations": int(adapter.nrelation),
        "score_dtype": "float32",
        "score_direction": "higher_is_better",
        "score_shape_per_mode": list(head_scores.shape),
        "exact_repeatability": {"head": repeat_head, "tail": repeat_tail},
        "architecture": adapter.architecture,
        "note": "ConvKB uses softplus(-pos) + softplus(neg) loss; larger scores mean more plausible triples.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/FB15k-237")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=Path("convkb_adapter_checks.json"))
    args = parser.parse_args()
    report = verify(args.checkpoint, args.data, args.device,
                    args.out.with_name("convkb_validation_score_examples.npz"))
    write_json(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Verify a checkpoint adapter against the upstream scorer on validation examples."""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from rotate.adapter import RotatEAdapter
from rotate.common import ROOT, file_hash, load_data, write_json
from vendor.kg_reeval.model import KGEModel


def method_hash(method):
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    def normalize(node):
        if isinstance(node, ast.AST):
            fields = dict(ast.iter_fields(node))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = fields["body"]
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    fields["body"] = body[1:]
            return {"node": type(node).__name__, "fields": {
                name: normalize(value) for name, value in fields.items()
                if name != "type_params"}}
        if isinstance(node, list):
            return [normalize(value) for value in node]
        return node
    # Empty type_params was added to FunctionDef in Python 3.12. Ignore it so
    # the same upstream code verifies in both Colab and local Python versions.
    canonical = json.dumps(normalize(tree), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def verify(checkpoint, data_dir, device="cpu", sample_path=None):
    splits, manifest = load_data(data_dir)
    adapter = RotatEAdapter.load(checkpoint, data_dir, device)
    source = json.loads((ROOT / "vendor/kg_reeval/SOURCE.json").read_text(encoding="utf-8"))
    for name in ("RotatE", "train_step"):
        if method_hash(getattr(KGEModel, name)) != source["unchanged_method_ast_sha256"][name]:
            raise AssertionError(f"Upstream {name} method has changed")
    triples = splits["valid"][:8]
    sampled = {}
    errors = {}
    candidates = np.random.default_rng(7).choice(adapter.nentity, size=32, replace=False)
    for mode in ("head", "tail"):
        scores = adapter.score_candidates(triples, mode)
        repeated = adapter.score_candidates(triples, mode)
        if scores.dtype != np.float32 or not np.array_equal(scores, repeated):
            raise AssertionError("Scores are not deterministic unrounded float32")
        max_error = 0.0
        for row, triple in enumerate(triples):
            target = triple[0 if mode == "head" else 2]
            for candidate in np.r_[candidates, target]:
                replacement = triple.copy()
                replacement[0 if mode == "head" else 2] = candidate
                with torch.inference_mode():
                    native = adapter.model(torch.as_tensor(replacement[None], dtype=torch.long,
                                                          device=adapter.device)).item()
                error = abs(float(scores[row, candidate]) - native)
                max_error = max(max_error, error)
                if not np.isclose(scores[row, candidate], native, atol=1e-4, rtol=1e-5):
                    raise AssertionError(f"Adapter/native mismatch for {mode}, query {row}, candidate {candidate}")
        sampled[f"scores_{mode}"] = scores
        errors[mode] = max_error
    if sample_path:
        np.savez_compressed(sample_path, triples=triples, **sampled)
    return {
        "passed": True, "checkpoint_sha256": file_hash(checkpoint), "device": str(adapter.device),
        "validation_triples_checked": len(triples), "entity_candidates_per_query": adapter.nentity,
        "relations": adapter.nrelation, "score_dtype": "float32", "score_direction": "higher_is_better",
        "score_shape_per_mode": [len(triples), adapter.nentity],
        "exact_repeatability": True, "mapping_hashes_match": True,
        "upstream_scoring_and_train_step_unchanged": True,
        "max_abs_error_vs_native_single": errors,
        "numerical_tolerance_note": "Head-batch uses an algebraically equivalent inverse rotation; tiny float32 differences from single mode are expected. Ties use the adapter scores without a tolerance.",
        "dataset_archive_sha256": manifest["source_archive_sha256"],
        "test_queries_scored": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/FB15k-237")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.checkpoint, args.data, args.device)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Build a portable ConvKB evaluator handoff without training data or optimizer state."""

import argparse
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from convkb.common import ROOT, file_hash, write_json
from convkb.verify import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/FB15k-237")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    if not summary["training_complete"] and not args.allow_partial:
        raise SystemExit("Training is incomplete. Resume it, or explicitly pass --allow-partial for a midway artifact.")
    if summary["name"] == "convkb_smoke" and not args.allow_smoke:
        raise SystemExit("Smoke weights are not an experimental result; pass --allow-smoke only to test handoff.")
    checkpoint = args.run / "best.pt"
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if metadata["step"] == 0 and not args.allow_smoke:
        raise SystemExit("Validation selected the untrained baseline. Train longer before handing this off as trained weights.")
    if args.output.exists() or args.output.with_suffix(".zip").exists():
        raise SystemExit("Handoff output already exists. Choose a new output path to avoid replacing an artifact.")
    args.output.mkdir(parents=True)
    report = verify(checkpoint, args.data, args.device,
                    args.output / "validation_score_examples.npz")
    shutil.copy2(checkpoint, args.output / "model.pt")
    for name in ("entities.dict", "relations.dict", "entity2id.txt", "relation2id.txt"):
        shutil.copy2(args.data / name, args.output / name)
    for name in ("config.json", "environment.json", "source_provenance.json", "requirements-lock.txt",
                 "validation_indices.json", "training.jsonl", "validation.jsonl", "summary.json"):
        if (args.run / name).exists():
            shutil.copy2(args.run / name, args.output / name)
    shutil.copy2(ROOT / "reports/data_manifest.json", args.output / "data_manifest.json")
    (args.output / "reports").mkdir()
    shutil.copy2(ROOT / "reports/data_manifest.json", args.output / "reports/data_manifest.json")
    shutil.copy2(ROOT / "prepare_data.py", args.output / "prepare_data.py")
    for folder in ("convkb", "rotate", "vendor"):
        shutil.copytree(ROOT / folder, args.output / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "docs/CONVKB_HANDOFF.md", args.output / "README.md")
    write_json(args.output / "adapter_checks.json", report)
    write_json(args.output / "handoff.json", {
        "model": "ConvKB", "selected_step": metadata["step"], "training_complete": summary["training_complete"],
        "smoke_only": summary["name"] == "convkb_smoke", "configuration": summary["name"],
        "selection": metadata["selection"], "checkpoint_sha256": file_hash(args.output / "model.pt"),
        "score_direction": "higher_is_better",
        "score_precision": "Raw float32 scores; no rounding, sigmoid, softmax, or mixed precision at scoring time",
        "candidate_order": "columns 0..14540 are entity IDs in entities.dict",
        "split_hash_reference": "data_manifest.json", "filtering_owner": "shared evaluator",
        "test_queries_scored": 0,
        "cross_device_note": "Evaluate each checkpoint once on one recorded device and reuse scores for all tie policies. CPU and GPU can differ at float32 precision.",
        "implementation_note": "PyTorch ConvKB adapted from daiquocnguyen/ConvKB; TensorFlow version in kg-reeval is not used. This deviation is documented in the report.",
    })
    checksums = {str(path.relative_to(args.output)).replace("\\", "/"): file_hash(path)
                 for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "SHA256SUMS.json", checksums)
    archive_path = args.output.with_suffix(".zip")
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(args.output)))
    checksum = file_hash(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    print(f"HANDOFF READY: {archive_path}")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
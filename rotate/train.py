"""Seeded RotatE training with validation-only selection and resumable batches."""

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from rotate.adapter import RotatEAdapter
from rotate.common import ROOT, environment, file_hash, load_data, source_provenance
from rotate.common import truth_indexes, validation_metrics, write_json
from vendor.kg_reeval.dataloader import TrainDataset
from vendor.kg_reeval.model import KGEModel


class TrainingBatches:
    """Single-process shuffled epochs; explicit state allows exact batch resume."""

    def __init__(self, triples, nentity, nrelation, batch_size, negative_size):
        self.datasets = {
            mode: TrainDataset([tuple(row) for row in triples.tolist()], nentity, nrelation,
                               negative_size, f"{mode}-batch")
            for mode in ("head", "tail")
        }
        self.batch_size = batch_size
        self.step = 0
        self.permutations = {mode: torch.empty(0, dtype=torch.long) for mode in self.datasets}
        self.positions = {mode: 0 for mode in self.datasets}

    def __next__(self):
        self.step += 1
        mode = "tail" if self.step % 2 else "head"
        dataset = self.datasets[mode]
        position, permutation = self.positions[mode], self.permutations[mode]
        if position >= len(permutation):
            permutation = torch.randperm(len(dataset))
            self.permutations[mode], position = permutation, 0
        indices = permutation[position:position + self.batch_size].tolist()
        self.positions[mode] = position + len(indices)
        return TrainDataset.collate_fn([dataset[index] for index in indices])

    def state_dict(self):
        return {"step": self.step, "permutations": self.permutations, "positions": self.positions}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.permutations = state["permutations"]
        self.positions = state["positions"]


def train_batch(model, optimizer, batch, microbatch_size, temperature):
    """Upstream weighted self-adversarial loss, accumulated over microbatches."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    positive, negative, weights, mode = batch
    device = model.entity_embedding.device
    positive, negative, weights = positive.to(device), negative.to(device), weights.to(device)
    denominator = weights.sum()
    total = {"loss": 0.0, "positive_sample_loss": 0.0, "negative_sample_loss": 0.0}
    for start in range(0, len(positive), microbatch_size):
        end = start + microbatch_size
        negative_scores = model((positive[start:end], negative[start:end]), mode=mode)
        negative_logp = (F.softmax(negative_scores * temperature, dim=1).detach()
                         * F.logsigmoid(-negative_scores)).sum(dim=1)
        positive_logp = F.logsigmoid(model(positive[start:end])).squeeze(dim=1)
        positive_loss = -(weights[start:end] * positive_logp).sum() / denominator
        negative_loss = -(weights[start:end] * negative_logp).sum() / denominator
        loss = (positive_loss + negative_loss) / 2
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss; checkpoint will not be updated")
        loss.backward()
        total["positive_sample_loss"] += positive_loss.item()
        total["negative_sample_loss"] += negative_loss.item()
        total["loss"] += loss.item()
    optimizer.step()
    return total


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_torch(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/FB15k-237")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Stop at this global step; not a completed training run")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. In Colab select Runtime > Change runtime type > T4 GPU.")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise SystemExit("Output directory is not empty. Use --resume or a new output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    splits, manifest = load_data(args.data)
    indexes = truth_indexes(splits)
    valid_indices = np.arange(len(splits["valid"]))
    if cfg["validation_triples"]:
        valid_indices = np.sort(np.random.default_rng(cfg["validation_seed"]).choice(
            len(valid_indices), size=min(cfg["validation_triples"], len(valid_indices)), replace=False))
    validation = splits["valid"][valid_indices]
    env = environment()
    provenance = source_provenance()
    model = KGEModel("RotatE", manifest["entities"], manifest["relations"],
                     cfg["hidden_dim"], cfg["gamma"], double_entity_embedding=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    batches = TrainingBatches(splits["train"], manifest["entities"], manifest["relations"],
                              cfg["batch_size"], cfg["negative_sample_size"])
    adapter = RotatEAdapter(model, cfg["candidate_chunk_size"])
    step, warm_up, rate, elapsed_before = 0, cfg["warm_up_steps"], cfg["learning_rate"], 0.0
    best_mrr, best_step, best_metrics = -1.0, None, None
    if args.resume:
        previous = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=True)
        if previous["config"] != cfg or previous["data_manifest"] != manifest:
            raise ValueError("Resume configuration or dataset differs from the saved run")
        if previous["source_provenance"]["file_sha256"] != provenance["file_sha256"]:
            raise ValueError("Training source changed; resume the original code or start a new run")
        if previous["environment"]["packages"] != env["packages"] or previous["environment"]["cuda_version"] != env["cuda_version"]:
            raise ValueError("Resume package/CUDA versions differ; restore the saved environment first")
        model.load_state_dict(previous["model_state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        batches.load_state_dict(previous["batches"])
        restore_rng(previous["rng"])
        step, warm_up, rate = previous["step"], previous["warm_up"], previous["learning_rate"]
        elapsed_before = previous["elapsed_seconds"]
        best_mrr, best_step, best_metrics = previous["best_mrr"], previous["best_step"], previous["best_metrics"]
    write_json(args.output / "validation_indices.json", valid_indices.tolist())
    write_json(args.output / "config.json", cfg)
    write_json(args.output / "environment.json", env)
    write_json(args.output / "source_provenance.json", provenance)
    (args.output / "requirements-lock.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in env["packages"].items()) + "\n",
        encoding="utf-8")
    started = time.perf_counter()
    peak_rss = psutil.Process().memory_info().rss
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def payload():
        return {"format_version": 1, "model": "RotatE", "config": cfg,
                "data_manifest": manifest, "source_provenance": provenance, "environment": env,
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "step": step, "score_direction": "higher_is_better", "score_dtype": "float32"}

    def validate():
        nonlocal best_mrr, best_step, best_metrics
        start = time.perf_counter()
        metrics = validation_metrics(adapter, validation, indexes, cfg["query_batch_size"])
        record = {"step": step, "split": "validation", "triples": len(validation),
                  "seconds": time.perf_counter() - start, **metrics}
        with (args.output / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("VALIDATION", json.dumps(record), flush=True)
        if metrics["MRR_expected_random"] > best_mrr:
            best_mrr, best_step, best_metrics = metrics["MRR_expected_random"], step, metrics
            save_torch(args.output / "best.pt", {**payload(), "selection": record})

    def save_latest():
        save_torch(args.output / "latest.pt", {
            **payload(), "optimizer_state_dict": optimizer.state_dict(), "batches": batches.state_dict(),
            "rng": rng_state(), "warm_up": warm_up, "learning_rate": rate,
            "best_mrr": best_mrr, "best_step": best_step, "best_metrics": best_metrics,
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        })

    if not args.resume:
        validate()
        save_latest()
    end_step = min(cfg["max_steps"], args.stop_after) if args.stop_after else cfg["max_steps"]
    print(f"Training {cfg['name']} on {args.device}; steps {step} -> {end_step}", flush=True)
    training_seconds = 0.0
    measured_steps = 0
    try:
        while step < end_step:
            tick = time.perf_counter()
            losses = train_batch(model, optimizer, next(batches), cfg["microbatch_size"],
                                  cfg["adversarial_temperature"])
            if args.device == "cuda":
                torch.cuda.synchronize()
            training_seconds += time.perf_counter() - tick
            measured_steps += 1
            step += 1
            # Upstream schedules against zero-based update indices and resets Adam.
            if step - 1 >= warm_up:
                rate /= 10
                optimizer = torch.optim.Adam(model.parameters(), lr=rate)
                warm_up *= 3
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            if step % cfg["log_every"] == 0:
                record = {"step": step, "learning_rate": rate, **losses,
                          "train_seconds_per_step": training_seconds / measured_steps,
                          "estimated_train_hours_remaining": (cfg["max_steps"] - step) * training_seconds / measured_steps / 3600}
                with (args.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print("TRAIN", json.dumps(record), flush=True)
            if step % cfg["validation_every"] == 0:
                validate()
            if step % cfg["checkpoint_every"] == 0:
                save_latest()
        if step % cfg["validation_every"]:
            validate()
        save_latest()
    except KeyboardInterrupt:
        # If interrupted during an update, resume from the most recent atomic checkpoint.
        print("Interrupted. Resume from latest.pt; the in-flight update was not saved.", flush=True)
        raise
    summary = {
        "name": cfg["name"], "description": cfg["description"], "completed_steps": step,
        "planned_steps": cfg["max_steps"], "training_complete": step >= cfg["max_steps"],
        "best_validation_step": best_step, "best_validation_metrics": best_metrics,
        "selection_split": "validation", "selection_triples": len(validation),
        "test_metrics_computed": False, "seed": cfg["seed"],
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
        "peak_process_rss_bytes_observed": peak_rss,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        "best_checkpoint_sha256": file_hash(args.output / "best.pt"),
        "environment": env,
    }
    write_json(args.output / "summary.json", summary)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

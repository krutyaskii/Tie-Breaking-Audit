"""Raw float32 scores for ConvKB. Higher is better; no inversion, no filtering."""

from pathlib import Path

import numpy as np
import torch

from convkb.common import file_hash
from convkb.model import ConvKB


class ConvKBAdapter:
    def __init__(self, model, candidate_chunk_size=2048):
        self.model = model.eval().float()
        self.nentity = model.nentity
        self.nrelation = model.nrelation
        self.chunk_size = candidate_chunk_size
        self.device = model.entity_embedding.weight.device

    @classmethod
    def load(cls, checkpoint, mapping_dir, device="cpu"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for name in ("entities.dict", "relations.dict"):
            if file_hash(Path(mapping_dir) / name) != payload["data_manifest"]["files"][name]["sha256"]:
                raise ValueError(f"Checkpoint mapping mismatch: {name}")
        cfg = payload["config"]
        model = ConvKB(payload["data_manifest"]["entities"],
                       payload["data_manifest"]["relations"],
                       cfg["hidden_size"], cfg["num_filters"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        adapter = cls(model, cfg["candidate_chunk_size"])
        adapter.metadata = payload
        return adapter

    @torch.inference_mode()
    def score_candidates(self, triples, mode):
        if mode not in ("head", "tail"):
            raise ValueError("mode must be 'head' or 'tail'")
        triples = torch.as_tensor(triples, dtype=torch.long, device=self.device)
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples must have shape [batch, 3], in head/relation/tail ID order")
        if triples.numel() and (
            torch.any(triples[:, [0, 2]] < 0) or torch.any(triples[:, [0, 2]] >= self.nentity)
            or torch.any(triples[:, 1] < 0) or torch.any(triples[:, 1] >= self.nrelation)
        ):
            raise ValueError("Entity or relation ID is outside the checkpoint's dictionary")
        B = len(triples)
        result = np.empty((B, self.nentity), dtype=np.float32)
        for start in range(0, self.nentity, self.chunk_size):
            end = min(start + self.chunk_size, self.nentity)
            candidates = torch.arange(start, end, device=self.device)
            C = candidates.size(0)
            if mode == "tail":
                heads = triples[:, 0].unsqueeze(1).expand(B, C)
                rels = triples[:, 1].unsqueeze(1).expand(B, C)
                tails = candidates.unsqueeze(0).expand(B, C)
            else:
                heads = candidates.unsqueeze(0).expand(B, C)
                rels = triples[:, 1].unsqueeze(1).expand(B, C)
                tails = triples[:, 2].unsqueeze(1).expand(B, C)
            raw = self.model(heads.reshape(-1), tails.reshape(-1), rels.reshape(-1)).reshape(B, C)
            result[:, start:end] = raw.cpu().numpy()
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite model scores")
        return result
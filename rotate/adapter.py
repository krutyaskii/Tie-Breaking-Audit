"""Raw float32 scores; larger is better. No filtering, sorting, or tie-breaking."""

from pathlib import Path

import numpy as np
import torch

from rotate.common import file_hash
from vendor.kg_reeval.model import KGEModel


class RotatEAdapter:
    def __init__(self, model, candidate_chunk_size=2048):
        if candidate_chunk_size < 1:
            raise ValueError('Candidate chunk size must be positive')
        self.model = model.eval().float()
        self.nentity = model.nentity
        self.nrelation = model.nrelation
        self.chunk_size = candidate_chunk_size
        self.device = model.entity_embedding.device

    @classmethod
    def load(cls, checkpoint, mapping_dir, device="cpu"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if any(v.is_floating_point() and v.dtype != torch.float32 for v in payload['model_state_dict'].values()):
            raise ValueError('Checkpoint weights must be float32')
        for name in ("entities.dict", "relations.dict"):
            if file_hash(Path(mapping_dir) / name) != payload["data_manifest"]["files"][name]["sha256"]:
                raise ValueError(f"Checkpoint mapping mismatch: {name}")
        cfg = payload["config"]
        model = KGEModel("RotatE", payload["data_manifest"]["entities"],
                         payload["data_manifest"]["relations"], cfg["hidden_dim"], cfg["gamma"],
                         double_entity_embedding=True)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        adapter = cls(model, cfg["candidate_chunk_size"])
        adapter.metadata = payload
        return adapter

    @torch.inference_mode()
    def score_candidates(self, triples, mode):
        if mode not in ("head", "tail"):
            raise ValueError("mode must be 'head' or 'tail'")
        triples = torch.as_tensor(triples, device=self.device)
        if triples.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError('Triples must contain integer IDs')
        triples = triples.long()
        if triples.ndim != 2 or triples.shape[1] != 3:
            raise ValueError("triples must have shape [batch, 3], in head/relation/tail ID order")
        if triples.numel() and (
            torch.any(triples[:, [0, 2]] < 0) or torch.any(triples[:, [0, 2]] >= self.nentity)
            or torch.any(triples[:, 1] < 0) or torch.any(triples[:, 1] >= self.nrelation)
        ):
            raise ValueError("Entity or relation ID is outside the checkpoint's dictionary")
        result = np.empty((len(triples), self.nentity), dtype=np.float32)
        for start in range(0, self.nentity, self.chunk_size):
            candidates = torch.arange(start, min(start + self.chunk_size, self.nentity), device=self.device)
            candidates = candidates.unsqueeze(0).expand(len(triples), -1).contiguous()
            scores = self.model((triples, candidates), mode=f"{mode}-batch")
            result[:, start:start + candidates.shape[1]] = scores.cpu().numpy()
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite model scores")
        return result

"""Scorer, mapping, and expected-metric checks for ConvKB."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from convkb.adapter import ConvKBAdapter
from convkb.common import file_hash, validation_metrics
from convkb.model import ConvKB


class ConvKBChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        np.random.seed(13)
        self.triples = np.array([(0, 0, 1), (0, 0, 2), (2, 1, 3), (1, 1, 0), (3, 0, 4)])

    def model(self):
        return ConvKB(5, 2, hidden_size=4, num_filters=4)

    def test_candidate_ids_match_native_single_scores(self):
        model = self.model()
        adapter = ConvKBAdapter(model, 3)
        for mode in ("head", "tail"):
            scores = adapter.score_candidates(self.triples, mode)
            for row, triple in enumerate(self.triples):
                for candidate in range(5):
                    replacement = triple.copy()
                    replacement[0 if mode == "head" else 2] = candidate
                    with torch.no_grad():
                        h, r, t = torch.tensor(replacement[None]).unbind(dim=1)
                        expected = model(h, t, r).item()
                    self.assertAlmostEqual(float(scores[row, candidate]), expected, places=5)
            np.testing.assert_array_equal(scores, adapter.score_candidates(self.triples, mode))

    def test_checkpoint_reload_and_mapping_mismatch(self):
        test_root = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="convkb_adapter_test_", dir=test_root) as temporary:
            root = Path(temporary)
            (root / "entities.dict").write_text("".join(f"{i}\te{i}\n" for i in range(5)), encoding="utf-8")
            (root / "relations.dict").write_text("0\tr0\n1\tr1\n", encoding="utf-8")
            model = self.model()
            torch.save({"config": {"hidden_size": 4, "num_filters": 4, "candidate_chunk_size": 3},
                        "data_manifest": {"entities": 5, "relations": 2,
                                          "files": {name: {"sha256": file_hash(root / name)}
                                                    for name in ("entities.dict", "relations.dict")}},
                        "model_state_dict": model.state_dict()}, root / "model.pt")
            restored = ConvKBAdapter.load(root / "model.pt", root)
            expected = ConvKBAdapter(model, 3).score_candidates(self.triples, "tail")
            np.testing.assert_array_equal(expected, restored.score_candidates(self.triples, "tail"))
            (root / "entities.dict").write_text("1\te0\n0\te1\n2\te2\n3\te3\n4\te4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping mismatch"):
                ConvKBAdapter.load(root / "model.pt", root)

    def test_validation_metrics_with_fixed_scores(self):
        class FixedScores:
            nentity = 5

            def score_candidates(self, triples, mode):
                return np.tile(np.array([3, 2, 2, 2, 1], dtype=np.float32), (len(triples), 1))

        indexes = ({(0, 1): {1, 2}}, {(1, 0): {1, 2}})
        result = validation_metrics(FixedScores(), np.array([[1, 0, 1]]), indexes, 1)
        self.assertAlmostEqual(result["MRR_expected_random"], (1 / 2 + 1 / 3) / 2)
        self.assertAlmostEqual(result["MRR_optimistic"], 1 / 2)
        self.assertAlmostEqual(result["MRR_pessimistic"], 1 / 3)
        self.assertEqual(result["Hits@1_expected_random"], 0)
        self.assertEqual(result["Hits@3_expected_random"], 1)
        self.assertEqual(result["queries"], 2)


if __name__ == "__main__":
    unittest.main()
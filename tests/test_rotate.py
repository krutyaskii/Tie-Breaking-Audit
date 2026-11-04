"""Independent scorer, loss, mapping, filtering, and resume checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rotate.adapter import RotatEAdapter
from rotate.common import file_hash, validation_metrics
from rotate.train import TrainingBatches, restore_rng, rng_state, train_batch
from vendor.kg_reeval.model import KGEModel


class RotatEChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        np.random.seed(13)
        self.triples = np.array([(0, 0, 1), (0, 0, 2), (2, 1, 3), (1, 1, 0), (3, 0, 4)])

    def model(self):
        return KGEModel("RotatE", 5, 2, 2, 6.0, double_entity_embedding=True)

    def test_manual_distance_and_score_direction(self):
        model = self.model()
        with torch.no_grad():
            model.relation_embedding.zero_()
            model.entity_embedding.copy_(torch.tensor([
                [1., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.],
                [0., 1., 0., 0.], [1., 0., 1., 0.]]))
        scores = RotatEAdapter(model, 2).score_candidates([[0, 0, 1]], "tail")[0]
        np.testing.assert_allclose(scores, [6., 6., 5., 4., 5.], atol=1e-6)
        self.assertEqual(scores.dtype, np.float32)

    def test_candidate_ids_match_native_single_scores_in_both_modes(self):
        model = self.model()
        adapter = RotatEAdapter(model, 3)
        for mode in ("head", "tail"):
            scores = adapter.score_candidates(self.triples, mode)
            for row, triple in enumerate(self.triples):
                for candidate in range(5):
                    replacement = triple.copy()
                    replacement[0 if mode == "head" else 2] = candidate
                    with torch.no_grad():
                        expected = model(torch.tensor(replacement[None])).item()
                    self.assertAlmostEqual(float(scores[row, candidate]), expected, places=5)
            np.testing.assert_array_equal(scores, adapter.score_candidates(self.triples, mode))

    def test_microbatch_update_matches_upstream_loss_and_optimizer(self):
        first, second = self.model(), None
        second = copy.deepcopy(first)
        batches = TrainingBatches(self.triples, 5, 2, 4, 3)
        batch = next(batches)
        optimizer_a = torch.optim.Adam(first.parameters(), lr=0.001)
        optimizer_b = torch.optim.Adam(second.parameters(), lr=0.001)
        args = SimpleNamespace(cuda=False, negative_adversarial_sampling=True,
                               adversarial_temperature=1.0, uni_weight=False, regularization=0.0)
        native = KGEModel.train_step(first, optimizer_a, iter([batch]), args)
        ours = train_batch(second, optimizer_b, batch, microbatch_size=2, temperature=1.0)
        self.assertAlmostEqual(native["loss"], ours["loss"], places=6)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], atol=2e-6, rtol=1e-5)

    def test_training_negatives_exclude_known_training_answers(self):
        batches = TrainingBatches(self.triples, 5, 2, 5, 32)
        for _ in range(4):
            positive, negative, _, mode = next(batches)
            known = set(map(tuple, self.triples.tolist()))
            for triple, candidates in zip(positive.tolist(), negative.tolist()):
                for candidate in candidates:
                    replacement = list(triple)
                    replacement[0 if mode == "head-batch" else 2] = candidate
                    self.assertNotIn(tuple(replacement), known)

    def test_sampler_and_rng_resume_exactly(self):
        batches = TrainingBatches(self.triples, 5, 2, 2, 4)
        next(batches)
        next(batches)
        state, random_state = copy.deepcopy(batches.state_dict()), rng_state()
        expected = [next(batches) for _ in range(5)]
        restored = TrainingBatches(self.triples, 5, 2, 2, 4)
        restored.load_state_dict(state)
        restore_rng(random_state)
        for reference in expected:
            actual = next(restored)
            self.assertEqual(reference[3], actual[3])
            for left, right in zip(reference[:3], actual[:3]):
                self.assertTrue(torch.equal(left, right))

    def test_filtered_random_mrr_is_average_of_reciprocal_ranks(self):
        class FixedScores:
            nentity = 5

            def score_candidates(self, triples, mode):
                return np.tile(np.array([3, 2, 2, 2, 1], dtype=np.float32), (len(triples), 1))

        # Entity 2 is another true answer and must be removed; target 1 must remain.
        indexes = ({(0, 1): {1, 2}}, {(1, 0): {1, 2}})
        result = validation_metrics(FixedScores(), np.array([[1, 0, 1]]), indexes, 1)
        self.assertAlmostEqual(result["MRR_expected_random"], (1 / 2 + 1 / 3) / 2)
        self.assertAlmostEqual(result["MRR_optimistic"], 1 / 2)
        self.assertAlmostEqual(result["MRR_pessimistic"], 1 / 3)
        self.assertEqual(result["Hits@1_expected_random"], 0)
        self.assertEqual(result["Hits@3_expected_random"], 1)
        self.assertEqual(result["queries"], 2)

    def test_checkpoint_reload_and_mapping_mismatch(self):
        test_root = Path(__file__).resolve().parents[1] / "artifacts" / "test_tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rotate_adapter_test_", dir=test_root) as temporary:
            root = Path(temporary)
            (root / "entities.dict").write_text("".join(f"{i}\te{i}\n" for i in range(5)), encoding="utf-8")
            (root / "relations.dict").write_text("0\tr0\n1\tr1\n", encoding="utf-8")
            model = self.model()
            torch.save({"config": {"hidden_dim": 2, "gamma": 6.0, "candidate_chunk_size": 3},
                        "data_manifest": {"entities": 5, "relations": 2,
                                          "files": {name: {"sha256": file_hash(root / name)}
                                                    for name in ("entities.dict", "relations.dict")}},
                        "model_state_dict": model.state_dict()}, root / "model.pt")
            restored = RotatEAdapter.load(root / "model.pt", root)
            expected = RotatEAdapter(model, 3).score_candidates(self.triples, "tail")
            np.testing.assert_array_equal(expected, restored.score_candidates(self.triples, "tail"))
            (root / "entities.dict").write_text("1\te0\n0\te1\n2\te2\n3\te3\n4\te4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping mismatch"):
                RotatEAdapter.load(root / "model.pt", root)


if __name__ == "__main__":
    unittest.main()

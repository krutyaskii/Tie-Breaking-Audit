"""Scorer, mapping, and expected-metric checks for ConvKB."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from convkb.adapter import ConvKBAdapter
from convkb.common import file_hash, validation_metrics
from convkb.model import ConvKB
from convkb.legacy_model import LegacyConvKB


class ConvKBChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        np.random.seed(13)
        self.triples = np.array([(0, 0, 1), (0, 0, 2), (2, 1, 3), (1, 1, 0), (3, 0, 4)])

    def model(self):
        return ConvKB(5, 2, hidden_size=4, num_filters=4)

    def test_independent_coordinate_formula(self):
        model = ConvKB(2, 1, hidden_size=4, num_filters=2)
        entity = np.array([[1,2,3,4],[4,3,2,1]],dtype=np.float32)
        relation = np.array([[.5,-1,1,2]],dtype=np.float32)
        kernels = np.array([[2,-1,3],[-1,2,-2]],dtype=np.float32)
        biases = np.array([-.5,1],dtype=np.float32)
        output = np.array([1,-2,3,-4,5,-6,7,-8],dtype=np.float32)
        with torch.no_grad():
            model.entity_embedding.weight.copy_(torch.from_numpy(entity))
            model.relation_embedding.weight.copy_(torch.from_numpy(relation))
            model.conv.weight.copy_(torch.from_numpy(kernels[:,None,:,None]))
            model.conv.bias.copy_(torch.from_numpy(biases))
            model.score_weight.copy_(torch.from_numpy(output[:,None]))
        for head,tail in [(0,1),(1,0)]:
            features = []
            for f in range(2):
                for i in range(4):
                    features.append(max(0,float(kernels[f,0]*entity[head,i]+kernels[f,1]*relation[0,i]+kernels[f,2]*entity[tail,i]+biases[f])))
            expected = sum(x*y for x,y in zip(features,output))
            actual = model(torch.tensor([head]),torch.tensor([tail]),torch.tensor([0])).item()
            self.assertAlmostEqual(actual,expected,places=6)
        self.assertEqual(tuple(model.conv.weight.shape),(2,1,3,1))

    def test_legacy_is_explicit(self):
        self.assertEqual(LegacyConvKB(5,2,4,4).architecture,'legacy_convkb_variant')
        model = self.model()
        with self.assertRaises(ValueError):
            ConvKBAdapter(model,0)
        with self.assertRaises(ValueError):
            ConvKBAdapter(model).score_candidates(np.array([[0.,0.,1.]]),'head')

    def test_training_objective_and_score_direction(self):
        from convkb.train import train_batch
        import torch.nn.functional as F
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(),lr=0)
        positive = torch.tensor([[0,0,1]])
        negative = torch.tensor([[2,3]])
        pos = model(positive[:,0],positive[:,2],positive[:,1])
        neg = model(torch.tensor([0,0]),negative.flatten(),torch.tensor([0,0]))
        expected = F.softplus(-pos).mean()+F.softplus(neg).mean()+.01*model.score_weight.square().sum()
        actual = train_batch(model,optimizer,(positive,negative,'tail'),.01,'cpu')
        self.assertAlmostEqual(actual['loss'],float(expected.detach()),places=6)
        score = torch.tensor([0.],requires_grad=True)
        F.softplus(-score).sum().backward()
        self.assertLess(score.grad.item(),0)  # minimizing loss raises positive scores


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
            legacy_payload = torch.load(root/'model.pt',weights_only=True)
            legacy_payload['model_state_dict'] = LegacyConvKB(5,2,4,4).state_dict()
            torch.save(legacy_payload,root/'legacy.pt')
            with self.assertRaisesRegex(ValueError,'architecture'):
                ConvKBAdapter.load(root/'legacy.pt',root)
            legacy = ConvKBAdapter.load(root/'legacy.pt',root,allow_legacy=True)
            self.assertEqual(legacy.architecture,'legacy_convkb_variant')
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

    def test_verifier_fails_on_tail_nonrepeatability(self):
        from unittest.mock import patch
        from convkb.verify import verify
        class Unstable:
            nentity, nrelation = 5,2
            calls = 0
            def score_candidates(self,triples,mode):
                self.calls += 1
                value = 1 if self.calls == 4 else 0
                return np.full((8,5),value,dtype=np.float32)
        with patch('convkb.verify.ConvKBAdapter.load',return_value=Unstable()), patch('convkb.verify.load_data',return_value=({'valid':np.zeros((8,3),dtype=np.int64)},{})):
            with self.assertRaisesRegex(ValueError,'repeatable'):
                verify('unused.pt','unused','cpu',Path('unused_examples.npz'))


if __name__ == "__main__":
    unittest.main()

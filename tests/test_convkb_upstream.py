"""Independent formula regression contributed in upstream commit 700e829."""
import unittest
import torch
from convkb.model import ConvKB


class ConvKBUpstreamChecks(unittest.TestCase):
    def test_forward_matches_manual_formula(self):
        torch.manual_seed(0)
        nentity, nrelation = 5, 2
        hidden_size, num_filters = 4, 3
        model = ConvKB(nentity, nrelation, hidden_size, num_filters)

        with torch.no_grad():
            model.entity_embedding.weight.copy_(torch.tensor([
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
                [1.7, 1.8, 1.9, 2.0],
            ]))
            model.relation_embedding.weight.copy_(torch.tensor([
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2],
            ]))
            model.conv.weight.copy_(torch.tensor([
                [[[0.1], [0.2], [0.3]]],
                [[[0.4], [0.5], [0.6]]],
                [[[0.7], [0.8], [0.9]]],
            ]))  # shape [3, 1, 3, 1]
            model.conv.bias.copy_(torch.tensor([0.01, 0.02, 0.03]))
            model.score_weight.copy_(torch.arange(1, 1 + num_filters * hidden_size, dtype=torch.float32).reshape(-1, 1) * 0.01)

        h, r, t = 0, 0, 1
        with torch.no_grad():
            actual = model(torch.tensor([h]), torch.tensor([t]), torch.tensor([r])).item()

        vh = model.entity_embedding.weight[h]
        vr = model.relation_embedding.weight[r]
        vt = model.entity_embedding.weight[t]
        total = 0.0
        for f in range(num_filters):
            wf = model.conv.weight[f, 0, :, 0]
            bf = model.conv.bias[f].item()
            for i in range(hidden_size):
                pre = (wf[0] * vh[i] + wf[1] * vr[i] + wf[2] * vt[i] + bf).item()
                post = max(pre, 0.0)  # ReLU
                sw = model.score_weight[f * hidden_size + i, 0].item()
                total += sw * post
        self.assertAlmostEqual(actual, total, places=5)

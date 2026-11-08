"""PyTorch ConvKB scorer, adapted from daiquocnguyen/ConvKB (Apache-2.0).

Score direction: higher is better. The upstream ConvKB loss
(softplus(-pos) + softplus(neg)) trains positive triples to have larger
scores than negative triples, so the adapter returns raw scores unchanged.
"""

import torch
import torch.nn as nn


class ConvKB(nn.Module):
    def __init__(self, nentity, nrelation, hidden_size=100, num_filters=128):
        super().__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.entity_embedding = nn.Embedding(nentity, hidden_size)
        self.relation_embedding = nn.Embedding(nrelation, hidden_size)
        # Conv2d with kernel (1, 3): 3 corresponds to the concatenated [h, r, t].
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(1, 3), bias=True)
        # After conv with kernel (1, 3), the hidden dimension shrinks by 2.
        self.score_weight = nn.Parameter(
            torch.empty(num_filters * 3 * (hidden_size - 2), 1)
        )
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)
        nn.init.normal_(self.score_weight, mean=0.0, std=0.1)

    def forward(self, head, tail, relation):
        """head, tail, relation: LongTensor [batch]. Returns [batch] scores."""
        h = self.entity_embedding(head)
        t = self.entity_embedding(tail)
        r = self.relation_embedding(relation)
        stacked = torch.stack([h, r, t], dim=1).unsqueeze(1)  # [batch, 1, 3, hidden]
        x = self.conv(stacked)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return (x @ self.score_weight).squeeze(-1)

    def forward_tail_batch(self, head, relation, candidates):
        batch = candidates.size(0)
        return self.forward(head.expand(batch), candidates, relation.expand(batch))

    def forward_head_batch(self, relation, tail, candidates):
        batch = candidates.size(0)
        return self.forward(candidates, tail.expand(batch), relation.expand(batch))
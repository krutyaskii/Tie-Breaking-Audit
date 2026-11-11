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
        # One filter combines h[i], r[i], t[i] for each coordinate i.
        self.conv = nn.Conv2d(
            1,
            num_filters,
            kernel_size=(3, 1),
            bias=True,
        )
        self.score_weight = nn.Parameter(
            torch.empty(num_filters * hidden_size, 1)
        )
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        nn.init.trunc_normal_(self.conv.weight, mean=0.0, std=0.1, a=-0.2, b=0.2)
        nn.init.constant_(self.conv.bias, 0.1)
        nn.init.normal_(self.score_weight, mean=0.0, std=0.1)

    def forward(self, head, tail, relation):
        """head, tail, relation: LongTensor [batch]. Returns [batch] scores."""
        h = self.entity_embedding(head)
        t = self.entity_embedding(tail)
        r = self.relation_embedding(relation)
        stacked = torch.stack([h, r, t], dim=1).unsqueeze(1)  # [B, 1, 3, hidden]
        x = torch.relu(self.conv(stacked))                    # [B, num_filters, 1, hidden]
        x = x.flatten(start_dim=1)                            # [B, num_filters * hidden]
        return (x @ self.score_weight).squeeze(-1)            # [B]

    def forward_tail_batch(self, head, relation, candidates):
        batch = candidates.size(0)
        return self.forward(head.expand(batch), candidates, relation.expand(batch))

    def forward_head_batch(self, relation, tail, candidates):
        batch = candidates.size(0)
        return self.forward(candidates, tail.expand(batch), relation.expand(batch))
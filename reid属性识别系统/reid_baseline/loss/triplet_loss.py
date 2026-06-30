import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        pairwise_dist = self._pairwise_distances(embeddings)

        anchor_positive = pairwise_dist.unsqueeze(2)
        anchor_negative = pairwise_dist.unsqueeze(1)

        triplet_loss = F.relu(
            anchor_positive - anchor_negative + self.margin
        )

        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        valid_triplet = mask.unsqueeze(2) * (1.0 - mask.unsqueeze(1))
        loss = (triplet_loss * valid_triplet).sum() / (valid_triplet.sum() + 1e-6)
        return loss

    def _pairwise_distances(self, x):
        dot_product = torch.matmul(x, x.T)
        square_norm = torch.diagonal(dot_product)
        distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
        return distances.clamp(min=1e-12).sqrt()

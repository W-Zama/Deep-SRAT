import torch
import torch.nn as nn


class PoissonLogLikelihoodLoss(nn.Module):
    def __init__(self):
        super(PoissonLogLikelihoodLoss, self).__init__()

    def forward(self, lambda_t, x):
        term1 = x * torch.log(lambda_t + 1e-2)
        term2 = lambda_t
        loss = -torch.sum(term1 - term2)
        return loss

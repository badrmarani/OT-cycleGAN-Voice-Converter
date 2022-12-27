import torch
from torch import nn


class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()


class CycleConsistencyLoss(nn.Module):
    def __init__(self):
        super(CycleConsistencyLoss).__init__()

    def forward(
        self,
    ):
        pass


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, disc_fake_y, disc_real_y):
        return (
            self.loss_fn((disc_fake_y), torch.ones_like(disc_real_y)) +
            self.loss_fn((disc_fake_y), torch.zeros_like(disc_real_y))
        )

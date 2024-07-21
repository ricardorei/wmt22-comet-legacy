import torch
from torch import nn

class AbstractLoss(object):

    @property
    def target_dim(self):
        raise NotImplementedError


class HeteroscedasticLoss(nn.Module):
    
    def forward(self, mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor):
        sigma = std**2
        log1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        mse = (target - mu)**2
        log2 = 0.5 * torch.log(sigma)
        return torch.sum(log1*mse+log2)
    
    @property
    def target_dim(self):
        return 2


class HeteroscedasticLossv2(nn.Module):
    
    def forward(self, mu: torch.Tensor, std: torch.Tensor, target: torch.Tensor):
        sigma = std**2
        log1 = 0.5 * torch.neg(torch.log(sigma)).exp() 
        mse = (target - mu)**2
        log2 = 0.5 * torch.log(sigma)
        return torch.sum(log1*mse+log2)

    @property
    def target_dim(self):
        return 2

class MSELoss(torch.nn.MSELoss, AbstractLoss):

    @property
    def target_dim(self):
        return 1

class KLLoss(torch.nn.Module, AbstractLoss):
    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target_mu: torch.Tensor,
        target_std: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.log(torch.abs(sigma) / torch.abs(target_std))
            + (target_std**2 + (target_mu - mu) ** 2) / (2 * sigma**2)
            - 0.5
        ).mean()

    @property
    def target_dim(self):
        return 2



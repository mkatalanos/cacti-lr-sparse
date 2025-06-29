from pytorch_msssim import MS_SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.ms_ssim_module = MS_SSIM(data_range=1, size_average=False)

    def forward(self, x, y):
        mse = F.mse_loss(x, y)

        # Normalize from [-1,1] to [0,1]
        x = (x+1)/2
        y = (y+1)/2

        ms_ssim_loss = 1 - self.ms_ssim_module(x, y)

        loss = self.alpha * mse + self.beta * ms_ssim_loss
        return loss

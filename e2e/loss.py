from pytorch_msssim import MS_SSIM
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.1, B=8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.ms_ssim_module = MS_SSIM(
            data_range=1, size_average=False, channel=B)

    def forward(self, x, y):
        mse = F.mse_loss(x, y)

        # Normalize from [-1,1] to [0,1]
        x = (x+1)/2
        y = (y+1)/2

        ms_ssim_loss = 1 - self.ms_ssim_module(x, y).mean()

        loss = self.alpha * mse + self.beta * ms_ssim_loss
        return loss


if __name__ == "__main__":
    from utils.visualize import visualize_cube

    def viz(tensor: torch.Tensor):
        visualize_cube(tensor.numpy().transpose(1, 2, 0))

    B, F_, M, N = 13, 8, 256, 256
    x = torch.randint(0, 100, (B, F_, M, N)).to(torch.float)

    # Generate Mask
    mask = torch.randint(0, 100, (B, F_, M, N)).to(torch.float)
    mask[mask <= 20] = 0
    mask[mask != 0] = 1

    y = torch.multiply(mask, x).sum(axis=1)

    # Invert
    mff = torch.multiply(mask, mask).sum(axis=1)
    mff[mff == 0] = 1e-8
    phiphit_inv = torch.divide(y, mff)

    inverted = torch.multiply(mask, phiphit_inv[:, torch.newaxis, :, :])

    xcuda = x.to("cuda")
    inv = inverted.to("cuda")

    loss = CustomLoss(B=F_)

    l = loss(xcuda, inv)

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
            data_range=255, size_average=False, channel=B)

    def forward(self, x, y):
        mse = F.mse_loss(x, y)

        # Normalize from [-1,1] to [0,255]
        x = (x+1)*127.5
        y = (y+1)*127.5

        ms_ssim_loss = 1 - self.ms_ssim_module(x, y).mean()

        loss = self.alpha * mse + self.beta * ms_ssim_loss
        return loss


if __name__ == "__main__":
    from utils.visualize import visualize_cube
    from model_wrapper import CustomModel
    from dataset import VideoDataset
    import torch.utils.data as data
    import glob

    def viz(tensor: torch.Tensor):
        visualize_cube(tensor.detach().cpu().numpy().transpose(1, 2, 0))

    data_dir = '/home/marios/Documents/diss-code/repo/e2e/dataset'
    sources = glob.glob(f"{data_dir}/*.mp4")
    sources = sources[:len(sources)*80//100]

    B = 20
    dataset = VideoDataset(sources, B, block_rate=0.2)
    dataloader = data.DataLoader(dataset, batch_size=4)

    model = CustomModel(B=B).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=model.lr)

    loss_fun = CustomLoss(B=B)

    for batch in dataloader:
        x = batch['truth'].to("cuda")
        y = batch['inverted'].to("cuda")

        pred = model(y)

        loss = loss_fun(x, pred)

        loss.backward()
        opt.step()
        opt.zero_grad()

        if y.isnan().any() or pred.isnan().any() or loss.isnan().any():
            print("Breaking")
            break
    #
    # loss = CustomLoss()
    #
    # l = loss(x, pred)

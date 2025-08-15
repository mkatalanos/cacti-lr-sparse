import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = out + x
        out = F.relu(out)
        return out


class E2E_CNN(nn.Module):
    def __init__(self, B=64, channels=64, depth=5):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Conv2d(B, channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.encoder = nn.ModuleList([
            ResBlock(channels) for i in range(depth)
        ])
        self.decoder = nn.ModuleList([
            ResBlock(channels) for i in range(depth)
        ])
        self.bridge = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )
        self.out_block = nn.Sequential(
            nn.Conv2d(channels, B, kernel_size=1),
            nn.Tanh()
        )
        self.depth = depth

    def forward(self, x):
        identity = x
        x = self.in_block(x)

        encoder_outputs = []

        # Encoder side
        for block in self.encoder:
            x = block(x)
            encoder_outputs.append(x)

        x = self.bridge(x)

        # Decoder
        for i, block in enumerate(self.decoder):
            skip_connection = encoder_outputs[-(i+1)]
            x = block(x+skip_connection)

        out = self.out_block(x)
        return out+identity


# if __name__ == "__main__":
#     model = E2E_CNN(B=3, channels=64, depth=5)
#     M, N, B = (256, 256, 3)
#     x = torch.randn((M, N, B))
#     x = x.permute(2, 0, 1)
#     out = model(x)
#     raise SystemExit

if __name__ == "__main__":
    from utils.visualize import visualize_cube
    from model_wrapper import CustomModel
    from dataset import VideoDataset
    import torch.utils.data as data
    import glob

    def viz(tensor: torch.Tensor):
        visualize_cube(tensor.transpose(1, 2, 0))

    data_dir = '/home/marios/Documents/diss-code/repo/e2e/dataset'
    sources = glob.glob(f"{data_dir}/*.mp4")[:3]

    B = 8
    dataset = VideoDataset(sources, B)
    dataloader = data.DataLoader(dataset, batch_size=4)

    for batch in dataloader:
        break

    truth = batch['truth']
    x = batch['inverted']

    channels = 64
    depth = 5

    in_block = nn.Sequential(
        nn.Conv2d(B, channels,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )

    encoder = nn.ModuleList([
        ResBlock(channels) for i in range(depth)
    ])

    decoder = nn.ModuleList([
        ResBlock(channels) for i in range(depth)
    ])

    bridge = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3,
                  stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3,
                  stride=1, padding=1),
        nn.ReLU(),
    )
    out_block = nn.Sequential(
        nn.Conv2d(channels, B, kernel_size=1),
        nn.Tanh()
    )

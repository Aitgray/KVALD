# Create the model
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# Define a simple U-Net for mask generation
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Encoder path
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # Decoder path
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # upsample
            skip = skip_connections[idx // 2]
            # pad in case of odd input dimensions
            if x.shape != skip.shape:
                x = nn.functional.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))


# Quick sanity check
if __name__ == "__main__":
    model = UNet(in_channels=1, out_channels=1)
    sample_input = torch.randn(1, 1, 256, 256)  # batch_size=1, grayscale 256×256
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expected: (1, 1, 256, 256)
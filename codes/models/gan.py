import torch
import torch.nn as nn

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

# Generator Network
class SRGenerator(nn.Module):
    def __init__(self, image_channels=3, feature_map_gen=64, num_res_blocks=8):
        super(SRGenerator, self).__init__()
        
        # Initial Convolution Layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(image_channels, feature_map_gen, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Residual Blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(feature_map_gen) for _ in range(num_res_blocks)])
        
        # Upsampling Layers (for super-resolution)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(feature_map_gen, feature_map_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_gen),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_gen, feature_map_gen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_gen),
            nn.ReLU(inplace=True)
        )
        
        # Final Output Layer
        self.final = nn.Sequential(
            nn.Conv2d(feature_map_gen, image_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        res = self.res_blocks(layer1)
        upsampled = self.upsample(res + layer1)  # SKIP connection from initial conv layer
        return self.final(upsampled)
        '''
            In an upsample network, a SKIP connection is added to preserve spatial information
        and prevent the loss of fine details during the upsampling process by directly feeding
        information from earlier layers in the encoder to the decoder, allowing the network
        to recover intricate features that might otherwise be lost due to the upsampling operation;
        this significantly improves the accuracy of the final output.
        '''


# Discriminator Network
class SRDiscriminator(nn.Module):
    def __init__(self, image_channels=3, feature_map_disc=64):
        super(SRDiscriminator, self).__init__()
        
        # Discriminator's Convolution Layers
        self.disc_net = nn.Sequential(
            nn.Conv2d(image_channels, feature_map_disc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_disc, feature_map_disc * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_disc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_disc * 2, feature_map_disc * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_disc * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_disc * 4, feature_map_disc * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_disc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_disc * 8, feature_map_disc * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_disc * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final Fully Connected Layer
        self.full_connect = nn.Sequential(
            nn.Linear(feature_map_disc * 16 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.disc_net(x)
        features = features.view(features.size(0), -1)  # Flatten before Fully Connected
        return self.full_connect(features)


import torch
import torch.nn as nn
import torchvision.models as tvmodels


class SuperResolutionLoss(nn.Module):
    def __init__(self, perceptual_weight=0.01):
        super(SuperResolutionLoss, self).__init__()
        self.pixel_loss = nn.L1Loss() # L1 Loss for pixel-wise comparison
        self.percecptual_weight = perceptual_weight

        # Load a pre-trained VGG network for perceptual loss
        vgg = tvmodels.vgg19(pretrained=True).features
        self.perceptual_network = nn.Sequential(*list(vgg[:16])).eval()  # Use layers up to relu4_1

        for params in self.perceptual_network:
            params.requires_grad_ = False   # Freeze VGG parameters
        

    def forward(self, gen_img, target_img):
        # Pixel wise loss
        pixel_loss = self.pixel_loss(gen_img, target_img)

        # Perceptual loss
        gen_features = self.perceptual_network(gen_img)
        target_features = self.perceptual_network(target_img)
        perceptual_loss = self.pixel_loss(gen_features, target_features)

        # Combine losses by giving different weight to perceptual loss
        total_loss = pixel_loss + perceptual_loss * self.percecptual_weight
        return total_loss

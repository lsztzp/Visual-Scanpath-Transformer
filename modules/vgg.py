import torch
import torchvision.models as models
from torch import nn

class VGG_Gen(nn.Module):
    def __init__(self, num_patch_h, num_patch_w, device):
        super(VGG_Gen, self).__init__()
        self.m = nn.UpsamplingBilinear2d(size=[num_patch_h, num_patch_w]).to(device)
        self.vgg = models.vgg19(pretrained=True, ).to(device)

    def forward(self, imgs, sals):
        imgs = self.vgg.features(imgs)
        imgs = self.m(imgs)
        enc_inputs = imgs.flatten(2).permute(0, 2, 1)  # [B, H ,W, C]
        return enc_inputs


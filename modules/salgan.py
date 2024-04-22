import torch
import numpy as np
import os
import torchvision
from torch import nn
from torchvision.transforms import transforms

conv_layer = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}


class Generator(nn.Module):
    def __init__(self, input_resize, pretrain=True):
        super(Generator, self).__init__()

        self.image_h = input_resize[0]
        self.image_w = input_resize[1]

        self.encoder = self.make_conv_layers(conv_layer['E'])
        self.decoder = self.make_deconv_layers(conv_layer['D'])

        self.net_params_path = 'model_weights/gen_modelWeights0090/'
        self.net_params_pathDir = os.listdir(self.net_params_path)
        self.net_params_pathDir.sort()
        self.mymodules = nn.ModuleList([
            self.deconv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        ])

        if pretrain:
            self.load_pretrain_weight()

    def upsampling(self, x):
        m = nn.UpsamplingBilinear2d(size=[self.image_h, self.image_w])
        x = m(x)
        return x

    def conv2d(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def deconv2d(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def relu(self, inplace=True):  # Change to True?
        return nn.ReLU(inplace)

    def maxpool2d(self, ):
        return nn.MaxPool2d(2)

    def make_conv_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [self.maxpool2d()]
            else:
                conv = self.conv2d(in_channels, v)
                layers += [conv, self.relu(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def make_deconv_layers(self, cfg):
        layers = []
        in_channels = 512
        for v in cfg:
            if v == 'U':
                layers += [nn.Upsample(scale_factor=2)]
            else:
                deconv = self.deconv2d(in_channels, v)
                layers += [deconv, self.relu(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def load_pretrain_weight(self,):
        print('salgan: pretrain-weights-loaded')
        params = self.state_dict()
        n1 = 0
        pretrained_dict = {}
        for k, v in params.items():
            single_file_name = self.net_params_pathDir[n1]
            single_file_path = os.path.join(self.net_params_path, single_file_name)
            pa = np.load(single_file_path)
            pa = torch.from_numpy(pa)
            pretrained_dict[k] = pa
            n1 += 1
        params.update(pretrained_dict)
        self.load_state_dict(params)

    def forward(self, x):
        x = self.encoder[0](x)
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        x = self.encoder[3](x)
        x = self.encoder[4](x)
        x = self.encoder[5](x)
        x = self.encoder[6](x)
        x = self.encoder[7](x)
        x = self.encoder[8](x)
        x = self.encoder[9](x)
        x = self.encoder[10](x)
        x = self.encoder[11](x)
        x = self.encoder[12](x)
        x = self.encoder[13](x)
        x = self.encoder[14](x)
        x = self.encoder[15](x)
        x = self.encoder[16](x)
        x = self.encoder[17](x)
        x = self.encoder[18](x)
        x = self.encoder[19](x)
        x = self.encoder[20](x)
        x = self.encoder[21](x)
        x = self.encoder[22](x)
        x = self.encoder[23](x)
        x = self.encoder[24](x)
        x = self.encoder[25](x)
        x = self.encoder[26](x)
        x = self.encoder[27](x)
        x = self.encoder[28](x)
        x = self.encoder[29](x)
        x = self.decoder[0](x)
        x = self.decoder[1](x)
        x = self.decoder[2](x)
        x = self.decoder[3](x)
        f6 = self.decoder[4](x)
        x = self.decoder[5](f6)
        x = self.decoder[6](x)
        x = self.decoder[7](x)
        x = self.decoder[8](x)
        x = self.decoder[9](x)
        x = self.decoder[10](x)
        f7 = self.decoder[11](x)
        x = self.decoder[12](f7)
        x = self.decoder[13](x)
        x = self.decoder[14](x)
        x = self.decoder[15](x)
        x = self.decoder[16](x)
        x = self.decoder[17](x)
        f8 = self.decoder[18](x)
        x = self.decoder[19](f8)
        x = self.decoder[20](x)
        x = self.decoder[21](x)
        x = self.decoder[22](x)
        f9 = self.decoder[23](x)
        x = self.decoder[24](f9)
        x = self.decoder[25](x)
        x = self.decoder[26](x)
        x = self.decoder[27](x)
        f10 = self.decoder[28](x)
        x = self.decoder[29](f10)

        x = self.mymodules[0](x)
        x = self.mymodules[1](x)
        # print(f6.shape, f10.shape)
        f6 = self.upsampling(f6).data
        f7 = self.upsampling(f7).data
        f8 = self.upsampling(f8).data
        f9 = self.upsampling(f9).data
        f10 = self.upsampling(f10).data

        return {
            'f6': f6,
            'f7': f7,
            'f8': f8,
            'f9': f9,
            'f10': f10,
            'x': x,
        }


class Feature_Extrator(nn.Module):
    def __init__(self, input_resize, feature_dim, patch_size, saliency_attention, sal_gen):
        super(Feature_Extrator, self).__init__()
        self.sal_gen = sal_gen
        self.saliency_attention = saliency_attention
        self.feature_dim = feature_dim
        self.net = Generator(input_resize)
        self.m = nn.BatchNorm2d(feature_dim, affine=True)
        self.pool2d = nn.AvgPool2d(patch_size)
        self.transform = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def forward(self, imgs, sals):
        images = self.transform(imgs).float()
        features = self.net(images)
        if self.sal_gen:
            sals = features['x']
        if self.feature_dim == 64:
            features = features['f10']
        elif self.feature_dim == 192:
            features = torch.cat((features['f9'], features['f10']), 1)  #
        elif self.feature_dim == 1472:
            features =  torch.cat((features['f6'], features['f7'], features['f8'], features['f9'], features['f10']), 1)  #
        elif self.feature_dim == 576:
            features =  torch.cat((features['f6'], features['f10']), 1)  #

        # print(features.shape)
        # features = Variable(torch.cat((images1, features), 1))
        features = self.m(features)  # [B, C ,H ,W]
        sals = sals.expand(imgs.size(0), self.feature_dim, features.size(2), features.size(3))
        if self.saliency_attention:
            features = features * sals
        features = self.pool2d(features)
        enc_inputs = features.flatten(2).permute(0, 2, 1)  # [B, H ,W, C]
        return enc_inputs

    def forward_sals_seqs(self, imgs, sals):
        sals = self.pool2d(sals)
        enc_inputs = sals.flatten(1).unsqueeze(-1)
        return enc_inputs

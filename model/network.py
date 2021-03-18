import typing as tp

import torch
import torch.nn as nn
import torch.nn.utils as spectral_norm

from model.AttentionLayers import MPGA, GLA


class ConvLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.01, *, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, X):
        return self.activation(self.conv(X))


class ConvLeakyReLUSpectralNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *, stride=2, padding=0, dilation=2):
        super().__init__()
        self.conv = ConvLeakyReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = spectral_norm(self.conv)

    def forward(self, X):
        return self.norm(self.conv(X))


def make_regular_branch():
    return nn.Sequential(ConvLeakyReLU(4, 32, 5),
                         ConvLeakyReLU(32, 32, 3, stride=2),
                         ConvLeakyReLU(32, 64, 3),
                         ConvLeakyReLU(64, 64, 3, stride=2),
                         ConvLeakyReLU(64, 128, 3, stride=2),
                         ConvLeakyReLU(128, 128, 3),
                         ConvLeakyReLU(128, 128, 3, dilation=2),
                         ConvLeakyReLU(128, 128, 3, dilation=4),
                         ConvLeakyReLU(128, 128, 3, dilation=8),
                         ConvLeakyReLU(128, 128, 3, dilation=16))


def make_decoder():
    return nn.Sequential(ConvLeakyReLU(256, 128, 3),
                         ConvLeakyReLU(128, 128, 3),
                         nn.Upsample(scale_factor=2, mode='nearest'),
                         ConvLeakyReLU(128, 64, 3),
                         ConvLeakyReLU(64, 64, 3),
                         nn.Upsample(scale_factor=2, mode='nearest'),
                         ConvLeakyReLU(64, 32, 3),
                         ConvLeakyReLU(32, 16, 3),
                         ConvLeakyReLU(16, 3, 3))


def make_attention_branch(attention_layer: tp.Union[MPGA, GLA]):
    return nn.Sequential(ConvLeakyReLU(4, 32, 5),
                         ConvLeakyReLU(32, 32, 3, stride=2),
                         ConvLeakyReLU(32, 64, 3, stride=2),
                         ConvLeakyReLU(64, 128, 3, stride=2),
                         ConvLeakyReLU(128, 128, 3, stride=2),
                         attention_layer,
                         ConvLeakyReLU(128, 128, 3),
                         ConvLeakyReLU(128, 128, 3))


class Encoder(nn.Module):
    def __init__(self, attention_layer: tp.Union[MPGA, GLA]):
        super().__init__()
        self.regular_branch = make_regular_branch()
        self.attention_branch = make_attention_branch(attention_layer)

    def forward(self, X):
        regular_branch_output = self.regular_branch(X)
        attention_branch_output = self.attention_branch(X)
        return torch.cat((regular_branch_output, attention_branch_output), dim=1)


class CoarseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = MPGA(dilatation=2)
        self.encoder = Encoder(self.attention_layer)
        self.decoder = make_decoder()

    def forward(self, X):
        return self.decoder(self.encoder(X))


class RefinementNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = GLA()
        self.encoder = Encoder(self.attention_layer)
        self.decoder = make_decoder()

    def forward(self, X):
        return self.decoder(self.encoder(X))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator_layers = nn.Sequential(ConvLeakyReLUSpectralNorm(3, 64, 5),
                                                  ConvLeakyReLUSpectralNorm(64, 128, 5),
                                                  ConvLeakyReLUSpectralNorm(128, 256, 5),
                                                  ConvLeakyReLUSpectralNorm(256, 256, 5),
                                                  ConvLeakyReLUSpectralNorm(256, 256, 5))
        self.

    def forward(self, inpainted_images, ground_truth_images):
        inpainted_feature_volumes = self.discriminator_layers(inpainted_images)
        ground_feature_volumes = self.discriminator_layers(ground_truth_images)
        

import typing as tp

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from model.AttentionLayers import MPGA, GLA


class ConvLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.01, *, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, X):
        #print('in convLeakyReLU', X.shape)
        return self.activation(self.conv(X))


class ConvLeakyReLUSpectralNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv_activate = ConvLeakyReLU(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation)
        self.normalized_conv = spectral_norm(self.conv_activate.conv)

    def forward(self, X):
        #print(X.shape)
        #return self.conv_activate(X)
        return self.normalized_conv(X)


def make_regular_branch():
    return nn.Sequential(ConvLeakyReLU(4, 32, 5, padding=2),
                         ConvLeakyReLU(32, 32, 3, stride=2, padding=1),
                         ConvLeakyReLU(32, 64, 3, padding=1),
                         ConvLeakyReLU(64, 64, 3, stride=2, padding=1),
                         ConvLeakyReLU(64, 128, 3, padding=1),
                         ConvLeakyReLU(128, 128, 3, padding=1),
                         ConvLeakyReLU(128, 128, 3, padding=2, dilation=2),
                         ConvLeakyReLU(128, 128, 3, padding=4, dilation=4),
                         ConvLeakyReLU(128, 128, 3, padding=8, dilation=8),
                         ConvLeakyReLU(128, 128, 3, padding=16, dilation=16))


def make_decoder():
    return nn.Sequential(ConvLeakyReLU(256, 128, 3, padding=1),
                         ConvLeakyReLU(128, 128, 3, padding=1),
                         nn.Upsample(scale_factor=2, mode='nearest'),
                         ConvLeakyReLU(128, 64, 3, padding=1),
                         ConvLeakyReLU(64, 64, 3, padding=1),
                         nn.Upsample(scale_factor=2, mode='nearest'),
                         ConvLeakyReLU(64, 32, 3, padding=1),
                         ConvLeakyReLU(32, 16, 3, padding=1),
                         nn.Conv2d(16, 3, 3, padding=1))


def make_attention_branch(attention_layer: tp.Union[MPGA, GLA]):
    return nn.Sequential(ConvLeakyReLU(4, 32, 5, padding=2),
                         ConvLeakyReLU(32, 32, 3, stride=2, padding=1),
                         ConvLeakyReLU(32, 64, 3, padding=1),
                         ConvLeakyReLU(64, 128, 3, stride=2, padding=1),
                         ConvLeakyReLU(128, 128, 3, padding=1),
                         attention_layer,
                         ConvLeakyReLU(128, 128, 3, padding=1),
                         ConvLeakyReLU(128, 128, 3, padding=1))


class AttentionBranch(nn.Module):
    def __init__(self, attention_layer):
        super().__init__()
        self.pre_attention = nn.Sequential(ConvLeakyReLU(4, 32, 5, padding=2),
                                           ConvLeakyReLU(32, 32, 3, stride=2, padding=1),
                                           ConvLeakyReLU(32, 64, 3, padding=1),
                                           ConvLeakyReLU(64, 128, 3, stride=2, padding=1),
                                           ConvLeakyReLU(128, 128, 3, padding=1))
        self.attention = attention_layer
        self.post_attention = nn.Sequential(ConvLeakyReLU(128, 128, 3, padding=1),
                                            ConvLeakyReLU(128, 128, 3, padding=1))

    def forward(self, image_mask, mask):
        feature_map = self.pre_attention(image_mask)
        attention_output = self.attention(feature_map, feature_map, mask)
        return self.post_attention(attention_output)


class Encoder(nn.Module):
    def __init__(self, attention_layer: tp.Union[MPGA, GLA]):
        super().__init__()
        self.regular_branch = make_regular_branch()
        self.attention_branch = AttentionBranch(attention_layer)

    def forward(self, image, mask):
        image_mask = torch.cat((image, mask), dim=1)
        regular_branch_output = self.regular_branch(image_mask)
        attention_branch_output = self.attention_branch(image_mask, mask)
        return torch.cat((regular_branch_output, attention_branch_output), dim=1)


class CoarseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = MPGA(in_channels=128)
        self.encoder = Encoder(self.attention_layer)
        self.decoder = make_decoder()

    def forward(self, image, mask):
        return torch.clamp(self.decoder(self.encoder(image, mask)), -1, 1)


class RefinementNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_layer = GLA(128)
        self.encoder = Encoder(self.attention_layer)
        self.decoder = make_decoder()

    def forward(self, course_image, mask):
        return torch.clamp(self.decoder(self.encoder(course_image, mask)), -1, 1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse = CoarseNetwork()
        self.refinement = RefinementNetwork()

    def forward(self, image, mask):
        course_image = self.coarse(image, mask)
        refinement_image = self.refinement(course_image, mask)
        return course_image, refinement_image


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator_layers = nn.Sequential(ConvLeakyReLUSpectralNorm(3, 64, 5, stride=2, padding=2),
                                                  ConvLeakyReLUSpectralNorm(64, 128, 5, stride=2, padding=2),
                                                  ConvLeakyReLUSpectralNorm(128, 256, 5, stride=2, padding=2),
                                                  ConvLeakyReLUSpectralNorm(256, 256, 5, stride=2, padding=2),
                                                  ConvLeakyReLUSpectralNorm(256, 256, 5, stride=2, padding=2))

    def forward(self, inpainted_images: torch.Tensor, ground_truth_images):
        n_channels = inpainted_images.shape[0]
        input_images = torch.cat((inpainted_images, ground_truth_images), dim=0)
        #print('D input shape', input_images.shape)
        # import ipdb; ipdb.set_trace()
        feature_volumes = self.discriminator_layers(input_images)
        inpainted_feature_volumes, ground_feature_volumes = torch.split(feature_volumes, n_channels, dim=0)
        return inpainted_feature_volumes, ground_feature_volumes

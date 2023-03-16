"""
Implements image encoders
"""

from collections import OrderedDict

from torch import profiler

from models.common.model.layers import *

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models
import torch.utils.model_zoo as model_zoo


# Code taken from https://github.com/nianticlabs/monodepth2
#
# Godard, Clément, et al.
# "Digging into self-supervised monocular depth estimation."
# Proceedings of the IEEE/CVF international conference on computer vision.
# 2019.

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            # x = self.convs[("upconv", i, 0)](x)
            x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                if x[0].shape[2] > input_features[i - 1].shape[2]:
                    x[0] = x[0][:, :, :input_features[i - 1].shape[2], :]
                if x[0].shape[3] > input_features[i - 1].shape[3]:
                    x[0] = x[0][:, :, :, :input_features[i - 1].shape[3]]
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            #x = self.convs[("upconv", i, 1)](x)
            x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

            self.outputs[("features", i)] = x

            if i in self.scales:
                #self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                self.outputs[("disp", i)] = self.sigmoid(self.decoder[self.decoder_keys[("dispconv", i)]](x))

        return self.outputs


class Decoder(nn.Module):
    def __init__(self, num_ch_enc, num_ch_dec=None, d_out=1, scales=range(4), use_skips=True):
        super(Decoder, self).__init__()

        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        if num_ch_dec is None:
            self.num_ch_dec = np.array([128, 128, 256, 256, 512])
        else:
            self.num_ch_dec = num_ch_dec
        self.d_out = d_out
        self.scales = scales

        self.num_ch_dec = [max(self.d_out, chns) for chns in self.num_ch_dec]

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.d_out)

        self.decoder_keys = {k: i for i, k in enumerate(self.convs.keys())}
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        with profiler.record_function("encoder_forward"):
            self.outputs = {}

            # decoder
            x = input_features[-1]
            for i in range(4, -1, -1):
                x = self.decoder[self.decoder_keys[("upconv", i, 0)]](x)

                x = [F.interpolate(x, scale_factor=(2, 2), mode="nearest")]

                if self.use_skips and i > 0:
                    feats = input_features[i - 1]

                    if x[0].shape[2] > feats.shape[2]:
                        x[0] = x[0][:, :, :feats.shape[2], :]
                    if x[0].shape[3] > feats.shape[3]:
                        x[0] = x[0][:, :, :, :feats.shape[3]]
                    x += [feats]
                x = torch.cat(x, 1)

                x = self.decoder[self.decoder_keys[("upconv", i, 1)]](x)

                self.outputs[("features", i)] = x

                if i in self.scales:
                    self.outputs[("disp", i)] = self.decoder[self.decoder_keys[("dispconv", i)]](x)

        return self.outputs


class Monodepth2(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        resnet_layers=18,
        cp_location=None,
        freeze=False,
        num_ch_dec=None,
        d_out=128,
        scales=range(4)
    ):
        super().__init__()

        self.encoder = ResnetEncoder(resnet_layers, True, 1)
        self.num_ch_enc = self.encoder.num_ch_enc

        self.upsample_mode = 'nearest'
        self.d_out = d_out
        self.scales = scales

        # decoder
        self.decoder = Decoder(num_ch_enc=self.num_ch_enc, d_out=self.d_out, num_ch_dec=num_ch_dec, scales=self.scales)
        self.num_ch_dec = self.decoder.num_ch_dec

        self.latent_size = self.d_out

        if cp_location is not None:
            cp = torch.load(cp_location)
            self.load_state_dict(cp["model"])

        if freeze:
            for p in self.parameters(True):
                p.requires_grad = False

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        with profiler.record_function("encoder_forward"):
            x = torch.cat([x * .5 + .5], dim=1)
            image_features = self.encoder(x)
            outputs = self.decoder(image_features)

            x = [outputs[("disp", i)] for i in self.scales]

        return x

    @classmethod
    def from_conf(cls, conf):
        return cls(
            cp_location=conf.get("cp_location", None),
            freeze=conf.get("freeze", False),
            num_ch_dec=conf.get("num_ch_dec", None),
            d_out=conf.get("d_out", 128),
            resnet_layers=conf.get("resnet_layers", 18)
        )

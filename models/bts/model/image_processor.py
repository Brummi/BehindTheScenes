from collections import namedtuple

import torch
from torch import nn
import lpips
import torch.nn.functional as F


def make_image_processor(config):
    type = config.get("type", "RGB").lower()
    if type == "rgb":
        ip = RGBProcessor()
    elif type == "perceptual":
        ip = PerceptualProcessor(config.get("layers", 1))
    elif type == "patch":
        ip = PatchProcessor(config.get("patch_size", 3))
    else:
        raise NotImplementedError(f"Unsupported image processor type: {type}")
    return ip


class RGBProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 3

    def forward(self, images):
        images = images * .5 + .5
        return images


class PerceptualProcessor(nn.Module):
    def __init__(self, layers=1) -> None:
        super().__init__()
        self.lpips_module = lpips.LPIPS(net="vgg")
        self._layers = layers
        self.channels = sum(self.lpips_module.chns[:self._layers])

    def forward(self, images):
        n, v, c, h, w = images.shape
        images = images.view(n*v, c, h, w)

        in_input = self.lpips_module.scaling_layer(images)

        x = self.lpips_module.net.slice1(in_input)
        h_relu1_2 = x
        x = self.lpips_module.net.slice2(x)
        h_relu2_2 = x
        x = self.lpips_module.net.slice3(x)
        h_relu3_3 = x

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        outs = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)

        feats = []

        for kk in range(self._layers):
            f = lpips.normalize_tensor(outs[kk])
            f = F.upsample(f, (h, w))
            feats.append(f)

        feats = torch.cat(feats, dim=1)

        feats = feats.view(n, v, self.channels, h, w)

        return feats


class PatchProcessor(nn.Module):
    def __init__(self, patch_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.channels = 3 * (patch_size ** 2)

        self._hps = self.patch_size // 2

    def forward(self, images):
        n, v, c, h, w = images.shape
        images = images.view(n*v, c, h, w) * .5 + .5

        images = F.pad(images, pad=(self.patch_size // 2,)*4, mode="replicate")
        h_, w_ = images.shape[-2:]

        parts = []

        for y in range(0, self.patch_size):
            for x in range(0, self.patch_size):
                parts.append(images[:, :, y:h_-(self.patch_size - y - 1), x:w_-(self.patch_size - x - 1)])

        patch_images = torch.cat(parts, dim=1)
        patch_images = patch_images.view(n, v, self.channels, h, w)

        return patch_images


class AutoMaskingWrapper(nn.Module):

    # Adds the corresponding color from the input frame for reference
    def __init__(self, image_processor):
        super().__init__()
        self.image_processor = image_processor

        self.channels = self.image_processor.channels + 1

    def forward(self, images, threshold):
        n, v, c, h, w = images.shape
        processed_images = self.image_processor(images)
        thresholds = threshold.view(n, 1, 1, h, w).expand(n, v, 1, h, w)
        processed_images = torch.stack((processed_images, thresholds), dim=2)
        return processed_images

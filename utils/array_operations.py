from functools import partial

import torch
import torch.nn.functional as F


unsqueezer = partial(torch.unsqueeze, dim=0)


def map_fn(batch, fn):
    if isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = map_fn(batch[k], fn)
        return batch
    elif isinstance(batch, list):
        return [map_fn(e, fn) for e in batch]
    else:
        return fn(batch)


def to(data, device, non_blocking=True):
    if isinstance(data, dict):
        return {k: to(data[k], device, non_blocking=non_blocking) for k in data.keys()}
    elif isinstance(data, list):
        return [to(v, device, non_blocking=non_blocking) for v in data]
    else:
        return data.to(device, non_blocking=non_blocking)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def mask_mean(t: torch.Tensor, m: torch.Tensor, dim=None, keepdim=False):
    t = t.clone()
    t[m] = 0
    els = 1
    if dim is None or len(dim)==0:
        dim = list(range(len(t.shape)))
    for d in dim:
        els *= t.shape[d]
    return torch.sum(t, dim=dim, keepdim=keepdim) / (els - torch.sum(m.to(torch.float), dim=dim, keepdim=keepdim))


def apply_crop(array, crop):
    return array[crop[0]:crop[0] + crop[2], crop[1]:crop[1] + crop[3]]


def shrink_mask(mask, shrink=3):
    mask = F.avg_pool2d(mask.to(torch.float32), kernel_size=shrink, padding=shrink // 2, stride=1)
    return (mask == 1.).to(torch.float32)


def get_mask(size, border=5, device=None):
    mask = torch.ones(size, dtype=torch.float32)
    mask = shrink_mask(mask, border)
    if device is not None:
        mask = mask.to(device)
    return mask


def get_grid(H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).flip(2).float() # flip h,w to x,y
    return grid


def detach(t):
    if isinstance(t, tuple):
        return tuple(t_.detach() for t_ in t)
    else: return t.detach()

import numpy as np
import torch

import matplotlib.pyplot as plt


def draw_bbox(im, size):
    b, c, h, w = im.shape
    h2, w2 = (h-size)//2, (w-size)//2
    marker = np.tile(np.array([[1.],[0.],[0.]]), (1,size))
    marker = torch.FloatTensor(marker)
    im[:, :, h2, w2:w2+size] = marker
    im[:, :, h2+size, w2:w2+size] = marker
    im[:, :, h2:h2+size, w2] = marker
    im[:, :, h2:h2+size, w2+size] = marker
    return im


def plot_image_grid(images, rows, cols, directions=None, imsize=(2, 2), title=None, show=True):
    fig, axs = plt.subplots(rows, cols, gridspec_kw={'wspace': 0, 'hspace': 0}, squeeze=True, figsize=(rows * imsize[0], cols * imsize[1]))
    for i, image in enumerate(images):
        axs[i % rows][i // rows].axis("off")
        if directions is not None:
            axs[i % rows][i // rows].arrow(32, 32, directions[i][0] * 16, directions[i][1] * 16, color='red', length_includes_head=True, head_width=2., head_length=1.)
        axs[i % rows][i // rows].imshow(image, aspect='auto')
    plt.subplots_adjust(hspace=0, wspace=0)
    if title is not None:
        fig.suptitle(title, fontsize=12)
    if show:
        plt.show()
    return fig


def show_save(save_path, show=True, save=False):
    if show:
        plt.show()
    if save:
        plt.savefig(save_path)


def color_tensor(tensor: torch.Tensor, cmap, norm=False):
    if norm:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    map = plt.cm.get_cmap(cmap)
    tensor = torch.tensor(map(tensor.cpu().numpy()), device=tensor.device)[..., :3]
    return tensor

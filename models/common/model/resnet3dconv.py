import torch
#  import torch_scatter
import torch.autograd.profiler as profiler
from torch import nn


# Resnet Blocks
class ResnetBlock3DConv(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, kernel_size=1, stride=1, padding=0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules

        self.conv_0 = nn.Conv3d(size_in, size_h, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_1 = nn.Conv3d(size_h, size_out, kernel_size=kernel_size, stride=stride, padding=padding)

        # Init
        nn.init.constant_(self.conv_0.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.conv_1.bias, 0.0)
        nn.init.zeros_(self.conv_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv3d(size_in, size_out, bias=False, kernel_size=kernel_size, stride=stride, padding=padding)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.conv_0(self.activation(x))
            dx = self.conv_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
            return x_s + dx


class Resnet3DConv(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=5,
        d_hidden=128,
        beta=0.0,
        kernel_size=1,
        stride=1,
        padding=0,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        if d_in > 0:
            self.conv_in = nn.Conv3d(d_in, d_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
            nn.init.constant_(self.conv_in.bias, 0.0)
            nn.init.kaiming_normal_(self.conv_in.weight, a=0, mode="fan_in")

        self.conv_out = nn.Conv3d(d_hidden, d_out, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.conv_out.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.blocks = nn.ModuleList(
            [ResnetBlock3DConv(d_hidden, beta=beta, kernel_size=kernel_size, stride=stride, padding=padding) for _ in range(n_blocks)]
        )

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, zx):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnet3dconv_infer"):
            x = zx
            if self.d_in > 0:
                x = self.conv_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            for blkid in range(self.n_blocks):
                x = self.blocks[blkid](x)
            out = self.conv_out(self.activation(x))
            return out

    @classmethod
    def from_conf(cls, conf, d_in, d_out, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            d_out,
            n_blocks=conf.get("n_blocks", 2),
            d_hidden=conf.get("d_hidden", 128),
            beta=conf.get("beta", 0.0),
            kernel_size=conf.get("kernel_size", 1),
            stride=conf.get("stride", 1),
            padding=conf.get("padding", 0),
            **kwargs
        )

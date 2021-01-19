import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_utils import split, merge, injective_pad, psi
import torchvision
import torchvision.transforms as transforms

def mean_dim(tensor, dim=None, keepdims=False):
    """Take the mean along multiple dimensions.
    Args:
        tensor (torch.Tensor): Tensor of values to average.
        dim (list): List of dimensions along which to take the mean.
        keepdims (bool): Keep dimensions rather than squeezing.
    Returns:
        mean (torch.Tensor): New tensor of mean value(s).
    """
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels,
                                 kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels,
                                  kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels,
                                  kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x


class ActNorm(nn.Module):
    """Activation normalization for 2D inputs.
    The bias and scale get initialized using the mean and variance of the
    first mini-batch. After the init, bias and scale are trainable parameters.
    Adapted from:
        > https://github.com/openai/glow
    """
    def __init__(self, num_features, scale=1., return_ldj=False):
        super(ActNorm, self).__init__()
        self.register_buffer('is_initialized', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.eps = 1e-6
        self.return_ldj = return_ldj

    def initialize_parameters(self, x):
        if not self.training:
            return

        with torch.no_grad():
            bias = -mean_dim(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = mean_dim((x.clone() + bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = (self.scale / (v.sqrt() + self.eps)).log()
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialized += 1.

    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs
        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()

        if sldj is not None:
            ldj = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - ldj
            else:
                sldj = sldj + ldj

        return x, sldj

    def forward(self, x, ldj=None, reverse=False):
        if not self.is_initialized:
            self.initialize_parameters(x)

        if reverse:
            x, ldj = self._scale(x, ldj, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, ldj = self._scale(x, ldj, reverse)

        if self.return_ldj:
            return x, ldj

        return x


class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.
    Args:
        num_channels (int): Number of channels in the input and output.
    """
    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        ldj = torch.slogdet(self.weight)[1] * x.size(2) * x.size(3)

        if reverse:
            weight = torch.inverse(self.weight.double()).float()
            sldj = sldj - ldj
        else:
            weight = self.weight
            sldj = sldj + ldj

        weight = weight.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weight)

        return z, sldj


class Coupling(nn.Module):
    """Affine coupling layer originally used in Real NVP and described by Glow.
    Note: The official Glow implementation (https://github.com/openai/glow)
    uses a different affine coupling formulation than described in the paper.
    This implementation follows the paper and Real NVP.
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate activation
            in NN.
    """
    def __init__(self, in_channels, mid_channels):
        super(Coupling, self).__init__()
        self.nn = NN(in_channels, mid_channels, 2 * in_channels)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))

    def forward(self, x, ldj, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, ldj


class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()
        # Activation normalization, invertible 1x1 convolution, affine coupling
        self.norm = ActNorm(in_channels, return_ldj=True)
        self.conv = InvConv(in_channels)
        self.coup = Coupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coup(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.norm(x, sldj, reverse)
        else:
            x, sldj = self.norm(x, sldj, reverse)
            x, sldj = self.conv(x, sldj, reverse)
            x, sldj = self.coup(x, sldj, reverse)

        return x, sldj

class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=1):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        out_ch = out_ch // 2
        self.stride = stride
        self.psi = psi(stride)

        layers = []
        if not first:
            layers.append(nn.BatchNorm2d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_ch//2, int(out_ch//mult), kernel_size=3,
                      stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)
        self.step = _FlowStep(in_ch, out_ch)

    def forward(self, x, sldj):
        """ bijective or injective block forward """
        x, sldj = self.step(x, sldj)
        x1, x2 = split(x)
        Fx2 = self.bottleneck_block(x2)
        y1 = Fx2 + x1
        return merge(x2, y1), sldj

    def inverse(self, x, sldj):
        """ bijective or injecitve block inverse """
        x2, y1 = split(x)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        x = merge(x1, x2)
        x, sldj = self.step.forward(x, sldj, reverse=True)
        return x, sldj


class Excessive(nn.Module):
    """This structure is trying to reimplement the MNIST experiment"""
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super(Excessive, self).__init__()
        self.ds = in_shape[2] // 2 ** (nStrides.count(2) + (len(nBlocks)) + init_ds // 2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2 ** self.init_ds
        self.nBlocks = nBlocks
        self.first = True
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch * 2 ** i for i in range(len(nBlocks))]
        self.init_psi = psi(self.init_ds)
        self.stack = nn.ModuleList()
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            self.stack.append(self.irevnet_stack(irevnet_block, channel, depth,
                                                 stride, dropout_rate=dropout_rate,
                                                 affineBN=affineBN, in_ch=self.in_ch,
                                                 mult=mult))
        self.bn1 = nn.BatchNorm2d(nChannels[-1] * 2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1] * 2, nClasses)

    def irevnet_stack(self, _block, channel, depth, stride, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = [stride] + [1] * (depth - 1)
        channels = [channel] * depth
        for channel, stride in zip(channels, strides):
            block_list.append(_block(channel, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))

            self.first = False
        return block_list

    def _preprocess(self, x):
        """Dequantize the input image `x`
        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
        Args:
            x (torch.Tensor): Input image.
        Returns:
            y (torch.Tensor): Dequantized logits of `x`.
        """
        # y = (x * 255. + torch.rand_like(x)) / 256.
        # y = (2 * y - 1) * self.bounds
        # y = (y + 1) / 2
        # y = y.log() - (1. - y).log()
        y = x
        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) \
              - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)
        return x, sldj
        # return y, sldj

    def forward(self, x):
        out_nui = []
        out, sldj = self._preprocess(x)
        out = squeeze(out)
        for stacks in self.stack:
            for block in stacks:
                out, sldj = block.forward(out, sldj)
            out = squeeze(out)
            out, out_2 = split(out)
            out_nui.append(out_2)
        out_bij = out
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out,self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_bij, out_nui, sldj

    def inverse(self, out_bij, out_nui, sldj):
        out = out_bij
        for i, stacks in enumerate(self.stack[::-1]):
            out = merge(out, out_nui[-(i+1)])
            out = squeeze(out, reverse=True)
            for block in stacks[::-1]:
                out, sldj = block.inverse(out, sldj)
        out = squeeze(out, reverse=True)
        return out


def squeeze(x, reverse=False):
    """Trade spatial extent for channels. In forward direction, convert each
    1x4x4 volume of input into a 4x1x1 volume of output.
    Args:
        x (torch.Tensor): Input to squeeze or unsqueeze.
        reverse (bool): Reverse the operation, i.e., unsqueeze.
    Returns:
        x (torch.Tensor): Squeezed or unsqueezed tensor.
    """
    b, c, h, w = x.size()
    if reverse:
        # Unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # Squeeze
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)
    return x


if __name__ == '__main__':
    batch=64
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])
    test_train = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch)
    exp = Excessive(nBlocks=[16,16,16], nStrides=[1,1,1], nClasses=10, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=(1,32,32), mult=1)
    x,_ = next(iter(testloader))
    print(x.shape)
    out, out_bij, out_nui, sldj = exp(x)
    print(out.shape)
    print(out_bij.shape)
    print(len(sldj))
    x_2 = exp.inverse(out_bij, out_nui, sldj)

    print(x_2-x)


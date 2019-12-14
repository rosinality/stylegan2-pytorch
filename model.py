import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, padding):
        ctx.save_for_backward(kernel)
        ctx.padding = kernel.shape[-1] - 1 - padding

        grad_input = F.conv_transpose2d(
            grad_output, kernel, padding=padding, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, = ctx.saved_tensors
        padding = ctx.padding

        grad_input = F.conv_transpose2d(
            gradgrad_output, kernel, padding=padding, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, padding):
        ctx.save_for_backward(kernel)
        ctx.padding = padding

        output = F.conv2d(input, kernel, padding=padding, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, = ctx.saved_tensors
        padding = ctx.padding

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, padding)

        return grad_input, None, None


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class Blur(nn.Module):
    def __init__(self, channel, filter, padding, factor=1):
        super().__init__()

        size = len(filter)
        self.padding = padding

        weight = torch.tensor(filter, dtype=torch.float32)
        weight = weight[:, None] * weight[None, :]
        weight = weight.view(1, 1, size, size)
        weight = weight / weight.sum() * (factor ** 2)

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return BlurFunction.apply(input, self.weight, self.padding)


class Upsample(nn.Module):
    def __init__(self, channel, filter):
        super().__init__()

        self.factor = 2
        self.pad0 = len(filter) // 2
        self.pad1 = (len(filter) - 1) // 2
        self.blur = Blur(channel, filter, padding=0, factor=self.factor)

    def forward(self, input):
        batch, channel, height, width = input.shape
        out = F.pad(
            input.view(batch, channel, height, 1, width, 1),
            [0, self.factor - 1, 0, 0, 0, self.factor - 1],
        )
        out = out.view(batch, channel, height * self.factor, width * self.factor)
        out = F.pad(out, [self.pad0, self.pad1, self.pad0, self.pad1])
        out = self.blur(out)

        return out


class Downsample(nn.Module):
    def __init__(self, channel, filter):
        super().__init__()

        self.factor = 2
        self.pad0 = len(filter) // 2
        self.pad1 = (len(filter) - 1) // 2
        self.blur = Blur(channel, filter, padding=0)

    def forward(self, input):
        batch, channel, height, width = input.shape
        out = F.pad(input, [self.pad0, self.pad1, self.pad0, self.pad1])
        out = self.blur(out)
        out = out[:, :, :: self.factor, :: self.factor]

        return out


class EqualLR:
    def __init__(self, name, lr_mul):
        self.name = name
        self.lr_mul = lr_mul

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = functools.reduce(operator.mul, weight.shape[1:])
        std = 1 / math.sqrt(fan_in)
        
        if self.lr_mul != 1:
            std *= self.lr_mul

        return weight * std

    @staticmethod
    def apply(module, name, lr_mul):
        fn = EqualLR(name, lr_mul)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight', lr_mul=1):
    EqualLR.apply(module, name, lr_mul)

    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=0, lr_mul=1):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim, bias=lr_mul == 1)
        linear.weight.data.normal_().div_(lr_mul)
        
        if lr_mul == 1:
            linear.bias.data.fill_(bias)
            self.bias = None

        else:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.linear = equal_lr(linear, lr_mul=lr_mul)

    def forward(self, input):
        out = self.linear(input)

        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        
        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            padding = (len(blur_kernel) - 2) - (kernel_size - 1)
            padding = (padding + 1) // 2 + 1

            self.blur = Blur(out_channel, blur_kernel, padding=padding, factor=2)

        if downsample:
            padding = (len(blur_kernel) - 2) + (kernel_size - 1)
            padding = padding // 2

            self.blur = Blur(in_channel, blur_kernel, padding=padding)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image):
        batch, _, height, width = image.shape

        noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = ScaledLeakyReLU(0.2)

    def forward(self, input, style):
        out = self.conv(input, style)
        out = self.noise(out)
        out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(3, blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            
            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01
    ):
        super().__init__()

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp))
            layers.append(ScaledLeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(channels[4])
        self.conv1 = StyledConv(
            channels[4], channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(channels[4], style_dim, upsample=False)

        log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = channels[4]

        for i in range(3, log_size + 1):
            out_channel = channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_layer = len(self.convs) + len(self.to_rgbs) + 1 + 1

    def forward(self, styles, return_latents=False):
        styles = [self.style(s) for s in styles]

        if len(styles) < 2:
            inject_index = self.n_layer

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            inject_index = random.randint(1, self.n_layer - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_layer - inject_index, 1)
            
            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 2

        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 3

        image = skip

        if return_latents:
            return image, latent
            
        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            padding = (len(blur_kernel) - 2) + (kernel_size - 1)
            padding = padding // 2

            layers.append(Blur(in_channel, blur_kernel, padding=padding))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias,
            )
        )

        if activate:
            layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False)
        
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        
        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]
        
        log_size = int(math.log(size, 2))
        
        in_channel = channels[size]
        
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            
            in_channel = out_channel
            
        self.convs = nn.Sequential(*convs)
        
        self.stddev_group = 4
        self.stddev_feat = 1
            
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, channels[4]),
                                          ScaledLeakyReLU(0.2),
                                          EqualLinear(channels[4], 1))
        
    def forward(self, input):
        out = self.convs(input)
        
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        
        return out
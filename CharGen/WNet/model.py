from impts import *

from dataType import DiscriminatorOutput, GeneratorOutput


_bn = [
     True,
     True,
     True,
     True,
     True,
     False
]

_ln = [
     False,
     True,
     True,
     True,
     True,
     True
]


class ConvBNRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int=5, pad: int=2, stride: int=2, bn: bool=False):
        """
        Initialize a conv-bn-relu block

        :param in_c: (int) is the size of input channel
        :param out_c: (int) is the size of output channel
        :param kernel_size: (int) is the size of kernel
        :param pad: (int) is the size for padding
        :param stride: (int) is the size of stride
        :param bn: (bool) whether to use batch-normalization here.
        """
        super().__init__()
        conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=pad, stride=stride)
        relu = nn.ReLU(inplace=False)
        self.main = nn.Sequential()
        if bn:
            self.main.add_module("conv", conv)
            self.main.add_module("bn", nn.BatchNorm2d(out_c))
            self.main.add_module("relu", relu)
        else:
            self.main.add_module("conv", conv)
            self.main.add_module("relu", relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class DeConvBNRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int=5, pad: int=2, stride: int=2, outpad: int=1, bn: bool=False):
        """
        Initialize a conv-bn-relu block

        :param in_c: (int) is the size of input channel
        :param out_c: (int) is the size of output channel
        :param kernel_size: (int) is the size of kernel
        :param pad: (int) is the size for padding
        :param stride: (int) is the size of stride
        :param bn: (bool) whether to use batch-normalization here.
        """
        super().__init__()
        #  Add outpadding such that the shape is correct
        conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel_size, padding=pad, stride=stride, output_padding=outpad)
        relu = nn.ReLU(inplace=False)
        self.main = nn.Sequential()
        if bn:
            self.main.add_module("conv", conv)
            self.main.add_module("bn", nn.BatchNorm2d(out_c))
            self.main.add_module("relu", relu)
        else:
            self.main.add_module("conv", conv)
            self.main.add_module("relu", relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class ConvLnLeakyReLU(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int=5, pad: int=2, stride: int=2, ln: bool=False, ln_w: int=0):
        """
        Initialize a conv-ln-lrelu block
        :param in_c: (int), input channel number
        :param out_c: (int), output channel
        :param kernel_size: (int) kernel_size: square
        :param pad: (int) padding
        :param stride: (int) stride
        :param ln: (bool) whether to use layer normalization
        :param ln_w: (int) the size of conv output (h, w), square
        """
        super().__init__()
        conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=pad, stride=stride)
        lrelu = nn.LeakyReLU(inplace=False, negative_slope=0.2)  # by the implementation in the paper
        self.main = nn.Sequential()
        if ln:
            self.main.add_module("conv", conv)
            self.main.add_module("ln", nn.LayerNorm([out_c, ln_w, ln_w]))
            self.main.add_module("relu", lrelu)
        else:
            self.main.add_module("conv", conv)
            self.main.add_module("relu", lrelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class BaseEncoder(nn.Module):
    """
    Is the prototype of the left/right part of the W-Net, maps 64x64x1 to 1x1x512 with conv-bn-layers
    """
    def __init__(self):
        super().__init__()
        bn = _bn  # Whether to use batch normalization or not
        # use padding=2 as the default value to make the size is n -> n/2
        # input 64x64x1 (Note in pytorch.conv2d, one needs the shape of N x Channel x H x W)
        self.conv0 = ConvBNRelu(1, 64, bn=bn[0])
        # 32x32x64
        self.conv1 = ConvBNRelu(64, 128, bn=bn[1])
        # 16x16x128
        self.conv2 = ConvBNRelu(128, 256, bn=bn[2])
        # 8x8x256
        self.conv3 = ConvBNRelu(256, 512, bn=bn[3])
        # 4x4x512
        self.conv4 = ConvBNRelu(512, 512, bn=bn[4])
        # 2x2x512
        self.conv5 = ConvBNRelu(512, 512, bn=bn[5])

        self.main = nn.ModuleList([self.conv0, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward propagation

        :param x:
        :return: a list of the tensor with size 5, return[i] is the result of (i+1)-th layer output
        """
        out = []  # tracing the forward propagation for further operation
        for mod in self.main:
            x = mod(x)
            out.append(x)
        return out


class ResBlock(nn.Module):
    """
    The residual block has the same structure as in:
        https://www.icst.pku.edu.cn/zlian/docs/20181024110234919639.pdf

    > each (residual block) consisting of two stacked BN-Relu-Conv architecture.
    By the w-net code, the conv is 3x3 stride 1 (need padding=1 to conserve the shape)
    """
    def __init__(self, chn: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(chn),
            nn.ReLU(inplace=False),
            nn.Conv2d(chn, chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(chn),
            nn.ReLU(inplace=False),
            nn.Conv2d(chn, chn, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x) -> torch.Tensor:
        return x + self.main(x)


class WNet(nn.Module):
    def __init__(self, M: int=5):
        super().__init__()
        self.EncP = BaseEncoder()  # Encoder prototype, left net
        self.EncR = BaseEncoder()  # Encoder reference, right net

        self.ResShortCutP0 = nn.Sequential(*[ResBlock(64) for _ in range(M-4)])
        self.ResShortCutP1 = nn.Sequential(*[ResBlock(128) for _ in range(M-2)])
        self.ResShortCutP2 = nn.Sequential(*[ResBlock(256) for _ in range(M)])
        self.ResShortCutR2 = nn.Sequential(*[ResBlock(256) for _ in range(M)])

        bn = _bn
        self.deconv5 = DeConvBNRelu(512 * 2, 512, bn=bn[5])  # feature concat
        self.deconv4 = DeConvBNRelu(512 * 3, 512, bn=bn[4])
        self.deconv3 = DeConvBNRelu(512 * 3, 256, bn=bn[3])
        self.deconv2 = DeConvBNRelu(256 * 3, 128, bn=bn[2])
        self.deconv1 = DeConvBNRelu(128 * 2, 64, bn=bn[1])
        self.deconv0 = DeConvBNRelu(64 * 2, 1, bn=bn[0])

    def forward(self, xP: torch.Tensor, xR: torch.Tensor) -> GeneratorOutput:
        lout = self.EncP(xP)
        rout = self.EncR(xR)

        r = self.deconv5(torch.cat([lout[5], rout[5]], dim=1))
        r = self.deconv4(torch.cat([lout[4], r, rout[4]], dim=1))
        r = self.deconv3(torch.cat([lout[3], r, rout[3]], dim=1))
        r = self.deconv2(torch.cat([self.ResShortCutP2(lout[2]), r, self.ResShortCutR2(rout[2])], dim=1))
        r = self.deconv1(torch.cat([self.ResShortCutP1(lout[1]), r], dim=1))
        r = self.deconv0(torch.cat([self.ResShortCutP0(lout[0]), r], dim=1))
        return GeneratorOutput(r, lout[-1], rout[-1])


# According to the paper, the discriminator is the composed by leakyReLU, layer_norm, and Conv
class Discriminator(nn.Module):
    def __init__(self, num_font=1, num_char=10):
        super().__init__()
        ln = _ln
        ddim = 32
        # input 64x64x1(HxWxC) (Note in pytorch.conv2d, one needs the shape of N x Channel x H x W)
        self.conv0 = ConvLnLeakyReLU(3, ddim, ln=ln[0], ln_w=32)
        # 32x32x64
        self.conv1 = ConvLnLeakyReLU(ddim, ddim * 2, ln=ln[1], ln_w=16)
        # 16x16x128
        self.conv2 = ConvLnLeakyReLU(ddim * 2, ddim * 4, ln=ln[2], ln_w=8)
        # 8x8x256
        self.conv3 = ConvLnLeakyReLU(ddim * 4, ddim * 8, ln=ln[3], ln_w=4)
        # 4x4x512
        self.conv4 = ConvLnLeakyReLU(ddim * 8, ddim * 16, ln=ln[4], ln_w=2)
        # 2x2x512
        self.conv5 = ConvLnLeakyReLU(ddim * 16, ddim * 32, ln=ln[5], ln_w=1)
        self.convs = nn.ModuleList([self.conv0, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5])

        self.fc_input_size = ddim * 32
        self.fc_gan = nn.Linear(ddim * 32, 1)  # real or fake
        self.fc_font = nn.Linear(ddim * 32, num_font)  # which font
        self.fc_char = nn.Linear(ddim * 32, num_char)

    def forward(self, x_g, x_p, x_r) -> DiscriminatorOutput:
        """
        Returns with:
            -  r_gan (Tensor), gan classification
            -  r_char (Tensor), char catagory classification
            -  r_font (Tensor), font catagory classification
            -  feat (list[Tensor]), features during conv, len=5

        :param x_g: (Tensor), generated character
        :param x_p: (Tensor), prototype character
        :param x_r: (Tensor), reference character
        :return: DiscriminatorOutput
        """
        x = torch.cat([x_p, x_g, x_r], dim=1)
        feat = []
        for mod in self.convs:
            x = mod(x)
            feat.append(x)

        x = x.view(-1, self.fc_input_size)
        r_gan = torch.sigmoid(self.fc_gan(x))
        r_font = F.softmax(self.fc_font(x), dim=1)
        r_char = F.softmax(self.fc_char(x), dim=1)
        return DiscriminatorOutput(r_gan, r_char, r_font, feat)


class FeatClassifierForEncoders(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


if __name__ == "__main__":
    d = Discriminator(2, 2)
    data = torch.randn(1, 1, 64, 64, dtype=torch.float)
    r1, r2, r3, _ = d(data, data, data)
    # print(data.size())
    print(r1, r2, r3)
    print(d(data, data, data))
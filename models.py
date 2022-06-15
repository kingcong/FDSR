import octconv as oc
import math
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
import mindspore.ops as ops


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=(kernel_size // 2), pad_mode='pad', has_bias=bias)

class MS_RB(nn.Cell):
    def __init__(self, num_feats, kernel_size):
        super(MS_RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=1,pad_mode = 'pad', dilation=1)
        self.conv2 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=kernel_size, padding=2,pad_mode = 'pad', dilation=2)
        self.conv4 = nn.Conv2d(in_channels=num_feats, out_channels=num_feats,
                               kernel_size=1, padding=0)
        self.act = nn.LeakyReLU()

    def construct(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x))
        x3 = x1 + x2
        x4 = self.conv4(x3)
        out = x4 + x

        return out

def resample_data(input, s):

    op = ops.Concat(axis=1)
    out = op(([input[:,:,i::s,j::s] for i in range(s) for j in range(s)]))

    return out
    
class PixelShuffle(nn.Cell):
    def __init__(self, block_size=2):
        super(PixelShuffle, self).__init__()
        self.block_size = block_size
        self.pixel_shuffle = P.DepthToSpace(block_size)

    def construct(self, x):
        return self.pixel_shuffle(x)


class Net(nn.Cell):
    def __init__(self, num_feats, depth_chanels, color_channel, kernel_size):
        super(Net, self).__init__()

        self.conv_rgb1 = nn.Conv2d(in_channels=48, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1,pad_mode='pad')
        self.rgb_cbl2 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0, alpha_out=0.25,
                                    stride=1, padding=1,pad_mode='pad',dilation=1,groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU())
        self.rgb_cbl3 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,pad_mode = 'pad',dilation=1,groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU())   
        self.rgb_cbl4 = oc.Conv_BN_ACT(in_channels=num_feats, out_channels=num_feats, kernel_size=kernel_size, alpha_in=0.25, alpha_out=0.25,
                                    stride=1, padding=1,pad_mode = 'pad',dilation=1,groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU())

        self.conv_dp1 = nn.Conv2d(in_channels=16, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1,pad_mode = 'pad')
        self.MSB1 = MS_RB(num_feats, kernel_size)
        self.MSB2 = MS_RB(56, kernel_size)
        self.MSB3 = MS_RB(80, kernel_size)
        self.MSB4 = MS_RB(104, kernel_size)
        self.conv_recon1 = nn.Conv2d(in_channels=104, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1,pad_mode = 'pad')
        self.ps1 = PixelShuffle(2)
        self.conv_recon2=nn.Conv2d(in_channels=num_feats, out_channels=4 * num_feats, kernel_size=kernel_size, padding=1,pad_mode = 'pad')
        self.ps2 = PixelShuffle(2)
        self.restore=nn.Conv2d(in_channels=num_feats, out_channels=1, kernel_size=kernel_size, padding=1,pad_mode = 'pad')
        self.act = nn.LeakyReLU()

    def construct(self, x):
        image, depth = x
        re_im = resample_data(image, 4)
        re_dp = resample_data(depth, 4)
        dp_in = self.act(self.conv_dp1(re_dp))
        dp1 = self.MSB1(dp_in)
        rgb1 = self.act(self.conv_rgb1(re_im))
        rgb2= self.rgb_cbl2(rgb1)
        op = ops.Concat(axis=1)
        ca1_in = op((dp1,rgb2[0]))
        dp2 = self.MSB2(ca1_in)
        rgb3 = self.rgb_cbl3(rgb2)
        ca2_in = op((dp2,rgb3[0]))
        dp3 = self.MSB3(ca2_in)
        rgb4 = self.rgb_cbl4(rgb3)
        ca3_in = op(([dp3,rgb4[0]]))
        dp4 = self.MSB4(ca3_in)
        up1 = self.ps1(self.conv_recon1(self.act(dp4)))
        up2 = self.ps2(self.conv_recon2(up1))
        out = self.restore(up2)
        out = depth + out
        return out
import math

import mindspore.nn as nn
import mindspore.ops as ops

class OctaveConv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1,pad_mode='pad',dilation=1,
                 groups=1):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = ops.ResizeBilinear((120,80))
        
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, stride = 1,padding=1, pad_mode = pad_mode, dilation=dilation, group = math.ceil(alpha_in * groups))
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, stride = 1, padding=padding, pad_mode = pad_mode, dilation=dilation,  group = groups)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, stride = 1, padding=padding, pad_mode = pad_mode, dilation=dilation, group = groups)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, stride = 1, padding=padding, pad_mode = pad_mode, dilation=dilation, group = math.ceil(groups - alpha_in * groups))

    def construct(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0,pad_mode='pad', dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding,pad_mode, dilation,
                               groups)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def construct(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class Conv_BN_ACT(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1,pad_mode='pad', dilation=1,
                 groups=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding,pad_mode, dilation,
                               groups)
        
        self.act = nn.LeakyReLU()
    def construct(self, x): 
        x_h, x_l = self.conv(x)
        
        x_h = self.act(x_h)
        x_l = self.act(x_l) if x_l is not None else None
        return x_h, x_l

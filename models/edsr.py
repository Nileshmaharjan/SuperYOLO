# from ..NN import common

import torch.nn as nn
import math 

def make_model(args, parent=False):
    return EDSR(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

# The Upsampler class inherits from nn.Sequential and is used to upscale an input image by a specified scale 
# factor using pixel shuffling. The __init__ method initializes a list of modules m that will be sequentially applied to the input tensor.
# If the specified scale factor is a power of 2, the method appends a sequence of convolutional layers and pixel shuffling operations to m
# that increase the number of channels by a factor of 4 at each step. If the scale factor is 3, the method appends a different set of convolutional 
# and pixel shuffling layers that increase the number of channels by a factor of 9. If the scale factor is neither a power of 2 nor 3, an error is raised.


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

# This code block defines a PyTorch module ResBlock that implements a residual block, which is commonly used in deep learning models for image processing tasks.
# A residual block consists of two convolutional layers with the same number of input and output channels, and a skip connection that adds the input tensor to the 
# output of the residual block.

# The __init__ method initializes a list of PyTorch modules m that are used to define the residual block. The input arguments to this method are as follows:

# conv: a convolutional layer that will be used to define the two convolutional layers in the residual block.
# n_feat: an integer specifying the number of input and output channels in the convolutional layers.
# kernel_size: an integer specifying the size of the kernel in the convolutional layers.
# bias: a boolean indicating whether or not to include bias terms in the convolutional layers.
# bn: a boolean indicating whether or not to use batch normalization after each convolutional layer.
# act: an activation function that will be applied after the first convolutional layer.
# res_scale: a scaling factor that is multiplied with the output of the residual block before it is added to the input tensor.
        
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        
        
#         The forward() method takes an input tensor x and applies the residual block to it. The method first passes x through the body of the residual block,
#         which consists of the two convolutional layers and the activation function if bn is True. The output of body is then multiplied by res_scale and added to
#         the input tensor x. The resulting tensor is returned as the output of the residual block.

# In the ResBlock module's forward() method, mul() is used to scale the output of the residual block by the res_scale factor before adding it to the input tensor x.

# For example, if res_scale is set to 0.1 and the output of the residual block is a tensor res, then res.mul(res_scale) 
# will multiply each element of res by 0.1. 
# The resulting tensor will be the same size as res but with each element scaled down by a factor of 0.1.

# By scaling the output of the residual block before adding it to the input tensor, the res_scale factor controls the strength 
# of the residual connection in the overall model.
# A higher res_scale value results in a stronger residual connection, whereas a lower value results in a weaker connection

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# The __init__ method defines the model architecture, with various hyperparameters that can be passed as arguments. Some key elements of the architecture include:

# n_resblock: number of residual blocks in the model.
# n_feats: number of feature maps in the model.
# kernel_size: size of the convolutional kernel used in the model.
# scale: factor by which to increase the resolution of the input image.
# act: activation function used in the model.
    
# The forward method defines the forward pass of the model. The input x is first passed through the head module, which applies a convolutional layer to the input. 
# The output of the head module is then passed through the body module, which consists of multiple ResBlock modules (residual blocks) that learn to 
# extract features from the input image. The output of the body module is added to the input x, which forms the residual connection in the model.
# The resulting tensor is then passed through the tail module, which upsamples the image using the Upsampler module and applies a final convolutional
# layer to generate the output image.
    
class EDSR(nn.Module):
    def __init__(self, num_channels=3,input_channel=64, factor=4, width=64, depth=16, kernel_size=3, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblock = depth
        n_feats = width
        kernel_size = kernel_size
        scale = factor
        act = nn.ReLU()

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(1.0, rgb_mean, rgb_std)

        # define head module
        m_head = [conv(input_channel, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1.
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, num_channels, kernel_size)
        ]

        # self.add_mean = common.MeanShift(1.0, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

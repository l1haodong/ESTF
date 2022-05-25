import torch
import torch.nn as nn
from model.sub_networks.context_net import ContextNet
from model.sub_networks.fpn import Offset_FPN_Concat
from model.sub_networks.our_blocks import SEBlock, CoordAtt
from model.sub_networks.grid_net import GridNet
from model.sub_networks.ppm import PPM
import functools
from math import sqrt

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        N, C, H, W = X.size()
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps)
        loss = torch.sum(error)
        loss = loss/(N*C*H*W)
        #print(loss)
        return loss


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))

# Net(nf=144, growth_rate=2, mode=poly)
class Net(nn.Module):
    def __init__(self, nf, growth_rate, mode):
        super(Net, self).__init__()

        nf = 64

        # Offset_FPN_Concat(48, growth_rate=2)
        #self.offset = Offset_FPN_Concat(nf//3, growth_rate)

        if mode == 'poly':
            from model.deformable.deform_conv3D_poly_3D import DeformConv3d as DCN3D
        elif mode == '3axis':
            from model.deformable.deform_conv3D_3Dinterpolation_inverse_distance import DeformConv3d as DCN3D
        elif mode == '1axis':
            from model.deformable.deform_conv3D_1DInterpolation_inverse_distance import DeformConv3d as DCN3D

        self.dcn_image = DCN3D(3, 3, nf, kernel_size=5, padding=2, stride=1, bias=True, modulation=True)
        self.dcn_context = DCN3D(6, 6, nf, kernel_size=5, padding=2, stride=1, bias=True, modulation=True)

        self.context = nn.Sequential(ContextNet(7, 16),
                                     CoordAtt(48, 3),        # 相当于一个注意力模块
                                     nn.Conv2d(48, 6, kernel_size=3, stride=1, padding=1)
                                     )
        #self.SE = SEBlock(9, 3)
        self.SE = CoordAtt(9, 3)
        self.grid = GridNet(9, depth_rate=16, growth_rate=16)

        if_bias = True
        conv1_1 = nn.Conv3d(
            3,
            32,
            kernel_size=(3,7,7),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)

        conv1_2 = nn.Conv3d(
            32,
            64,
            kernel_size=(3,3,3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)
        bn1_2 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)


        conv1_3 = nn.Conv3d(
        64,
        64,
        kernel_size=(3,3,3),
        stride=(1, 2, 2),
        padding=(0, 0, 0),
        bias=if_bias)
        
        bn1_3 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)
        
        pad1_1 = nn.ReplicationPad3d((3, 3, 3, 3, 1, 1))
        pad1_2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        pad1_3 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))

        downsample_model = [pad1_1, conv1_1, bn1_1, relu, pad1_2, conv1_2, bn1_2, relu, pad1_3, conv1_3, bn1_3, relu]
        #downsample_model = [conv1_1, bn1_1, relu, conv1_2, bn1_2, relu, conv1_3, bn1_3, relu]
        self.downsample_model = nn.Sequential(*downsample_model)

        self.SE1 = CoordAtt(64*4, 16)
        self.pre = nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1)

        deconv2_1 = nn.ConvTranspose3d(64,64,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        bn2_1 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)

        deconv2_2 = nn.ConvTranspose3d(64,64,kernel_size=(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=if_bias)
        bn2_2 = nn.BatchNorm3d(64)
        relu = nn.ReLU(inplace=True)

        conv2_2 = nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        bn2_3 = nn.BatchNorm3d(64)

        conv2_3 = nn.Conv3d(64,64,kernel_size=(3,7,7),stride=(1,1,1),padding=(0,0,0),bias=if_bias)
        tanh = nn.Tanh()

        pad2_2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        pad2_3 = nn.ReplicationPad3d((3, 3, 3, 3, 1, 1))
        upsample_model = [deconv2_1, bn2_1, relu, deconv2_2, bn2_2, relu, pad2_2, conv2_2, bn2_3, pad2_3, conv2_3, tanh]
        self.upsample_model = nn.Sequential(*upsample_model)

        self.nf2 = 64
        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, self.nf2), 5)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, img1, img2, img4, img5, img_name=None):

        x = self.downsample_model(torch.cat((img1.unsqueeze(2), img2.unsqueeze(2), img4.unsqueeze(2), img5.unsqueeze(2)), dim=2))
        # print(x.shape)  # torch.Size([1, 64, 4, 64, 64])

        x = self.residual_layer(x)
        offset_central = self.upsample_model(x)
        #B, N, C, H, W = offset_central.size()
        offset_central = torch.cat(torch.unbind(offset_central , 2) , 1)
        #offset_central = offset_central.reshape(B, N*C, H, W)

        offset_central = self.SE1(offset_central)
        offset_central = self.relu(self.pre(offset_central))
        
        #print("offset_central get")

        mid_out, g_Spatial_img = self.dcn_image(img1, img2, img4, img5, offset_central)
        #print("mid_out get")

        image1_context = self.context(img1)
        image2_context = self.context(img2)
        image4_context = self.context(img4)
        image5_context = self.context(img5)

        central_context,g_Spatial_ctx  = self.dcn_context(image1_context, image2_context,
                                           image4_context, image5_context, offset_central)
        #print("central_context get")

        out = self.SE(torch.cat((mid_out, central_context), dim=1))         # 级联之后，还要再加一个通道注意力，为啥？
        #print("out before")
        out = self.grid(out)
        #print("out after")
        if self.training:
            # 为什么out = out + mid_out？
            #print(g_Spatial_img)
            #print(g_Spatial_ctx)
            g_Spatial = g_Spatial_img + g_Spatial_ctx
            # print(g_Spatial)
            return out + mid_out, mid_out, g_Spatial
        else:
            return mid_out+out

from dcn.modules.deform_conv import *
class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()

        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

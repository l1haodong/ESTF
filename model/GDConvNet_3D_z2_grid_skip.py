import torch
import torch.nn as nn
from model.sub_networks.context_net import ContextNet
from model.sub_networks.fpn import Offset_FPN_Concat
from model.sub_networks.our_blocks import SEBlock, CoordAtt
from model.sub_networks.grid_net_enhanced import GridNet
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

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1

class FeatureExtraction(torch.nn.Module):
    def __init__(self):
        super(FeatureExtraction, self).__init__()
        if_bias = True
        self.conv1_1 = nn.Conv3d(
            3,
            32,
            kernel_size=(3, 7, 7),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=if_bias)
        self.bn1_1 = nn.BatchNorm3d(32)
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv3d(
            32,
            64,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)
        self.bn1_2 = nn.BatchNorm3d(64)
        self.relu1_2 = nn.ReLU(inplace=True)

        self.conv1_3 = nn.Conv3d(
            64,
            64,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 0, 0),
            bias=if_bias)

        self.bn1_3 = nn.BatchNorm3d(64)
        self.relu1_3 = nn.ReLU(inplace=True)

        self.pad1_1 = nn.ReplicationPad3d((3, 3, 3, 3, 1, 1))
        self.pad1_2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))
        self.pad1_3 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))

        self.deconv2_1 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=if_bias)
        self.bn2_1 = nn.BatchNorm3d(64)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.deconv2_2 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=if_bias)
        self.bn2_2 = nn.BatchNorm3d(64)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv2_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0), bias=if_bias)


        self.pad2_2 = nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))

        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, self.nf2), 5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        self.joinType = "concat"

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, img1, img2, img4, img5):

        x = torch.cat((img1.unsqueeze(2), img2.unsqueeze(2), img4.unsqueeze(2), img5.unsqueeze(2)), dim=2)

        tensor32_256 = self.relu1_1(self.bn1_1(self.conv1_1(self.pad1_1(x))))
        tensor64_128 = self.relu1_2(self.bn1_2(self.conv1_2(self.pad1_2(tensor32_256))))
        tensor64_64 = self.relu1_3(self.bn1_3(self.conv1_3(self.pad1_3(tensor64_128))))

        tensor64_64_2 = self.residual_layer(tensor64_64)

        tensor64_64_concat = joinTensors(tensor64_64, tensor64_64_2, type=self.joinType)
        tensor64_128_2 = self.relu2_1(self.bn2_1(self.deconv2_1(tensor64_64_concat)))
        tensor64_128_concat = joinTensors(tensor64_128, tensor64_128_2, type=self.joinType)
        tensor64_256_2 = self.relu2_2(self.bn2_2(self.deconv2_2(tensor64_128_concat)))
        offset_central = self.conv2_2(self.pad2_2(tensor64_256_2))
        return offset_central

# Net(nf=144, growth_rate=2, mode=poly)
class Net(nn.Module):
    def __init__(self, nf, growth_rate, mode):
        super(Net, self).__init__()
        nf = 64

        self.source_feature = FeatureExtraction()
        self.SE1 = CoordAtt(64 * 4, 16)
        self.pre = nn.Conv2d(64 * 4, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if mode == 'poly':
            from model.deformable.deform_conv3D_poly_3D_z2_grid_skip import DeformConv3d as DCN3D
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

        self.SE = CoordAtt(9, 3)

        self.grid = GridNet(9, depth_rate=16, growth_rate=16)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, img1, img2, img4, img5, img_name=None):
        offset_central = self.source_feature(img1, img2, img4, img5,)

        offset_central = torch.cat(torch.unbind(offset_central, 2), 1)

        offset_central = self.SE1(offset_central)
        offset_central = self.relu(self.pre(offset_central))

        mid_out, g_Spatial_img = self.dcn_image(img1, img2, img4, img5, offset_central)

        image1_context = self.context(img1)
        image2_context = self.context(img2)
        image4_context = self.context(img4)
        image5_context = self.context(img5)

        central_context,g_Spatial_ctx  = self.dcn_context(image1_context, image2_context,
                                           image4_context, image5_context, offset_central)

        out = self.SE(torch.cat((mid_out, central_context), dim=1))         # 级联之后，还要再加一个通道注意力，为啥？

        #print("out before")
        out = self.grid(out)
        #print("out after")

        if self.training:

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
class ResBlock(nn.Module):
    def __init__(self, nf):
        super(ResBlock, self).__init__()
        self.dcn0 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x
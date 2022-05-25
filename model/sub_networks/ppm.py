import torch.nn as nn
import torch.nn.functional as F
import torch

class PPM(nn.Module):

    def __init__(self):
        super(PPM, self).__init__()
        # self.trans = LinkNet50(n_classes=32)
        self.tanh = nn.Tanh()

        # self.refine0 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.refine1 = nn.Conv2d(9, 20, kernel_size=3, stride=1, padding=1)
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.threshold = nn.Threshold(0.1, 0.1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.refine3 = nn.Conv2d(20 + 4, 3, kernel_size=3, stride=1, padding=1)

        self.upsample = F.upsample_nearest

        self.relu0 = nn.LeakyReLU(0.2)
        self.relu1 = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.relu3 = nn.LeakyReLU(0.2)
        self.relu4 = nn.LeakyReLU(0.2)
        self.relu5 = nn.LeakyReLU(0.2)
        self.relu6 = nn.LeakyReLU(0.2)

    def forward(self, I):
        # t = self.trans(I)

        # Adapted from He Zhang https://github.com/hezhangsprinter/DCPDN
        # Bring I to feature space for concatenation
        # I = self.relu0((self.refine0(I)))
        # dehaze = torch.cat([t, I], 1)

        dehaze = self.relu1((self.refine1(I)))
        dehaze = self.relu2((self.refine2(dehaze)))

        shape_out = dehaze.data.size()
        shape_out = shape_out[2:4]
        x101 = F.avg_pool2d(dehaze, 32)
        # x1010 = F.avg_pool2d(dehaze, 32)
        x102 = F.avg_pool2d(dehaze, 16)
        # x1020 = F.avg_pool2d(dehaze, 16)
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = self.upsample(self.relu3(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu4(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu5(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu6(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze = self.tanh(self.refine3(dehaze))
        return dehaze

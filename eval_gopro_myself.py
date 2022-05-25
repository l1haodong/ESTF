import os
import time
# from util.utils_test import to_psnr
import pytorch_ssim

from torchvision.utils import save_image as imwrite
from configs.config_test import device_id, mode, ValData, val_batch_size, model_save_path
import torch
from model.GDConvNet import Net
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
import numpy as np
from lpips_pytorch import lpips
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision.transforms as TF

def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])
def to_loss(rec, gt):
    mse = F.mse_loss(rec, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    return mse_list
from math import log10
def to_psnr(rec, gt):
    mse = F.mse_loss(rec, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    try:
        psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    except:
        print(mse_list)
    return psnr_list


all_start_time = time.time()

device_ids = device_id
device = torch.device("cuda:{}".format(device_id[0]) if torch.cuda.is_available() else "cpu")


# Build model
net = Net(144, growth_rate=2, mode=mode)


# multi-GPU
net = net.to(device)
print(device_ids)
net = nn.DataParallel(net, device_ids=device_ids)

# print('===============================')
# print("# of model parameters by CDFI is: " + str(count_network_parameters(net)))

# calculate all trainable parameters in network
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params by GDConv: {}".format(pytorch_total_params))

# print("#params by FLAVR" , sum([p.numel() for p in net.parameters()]))

net.load_state_dict(torch.load('/data/haodongli/GDConvNet-ori/modeldict/poly/net_best_weight', map_location='cuda:0'), strict=False)




# val_data_loader_full = DataLoader(ValData(), batch_size=val_batch_size, shuffle=False, num_workers=12)
# # val_data_loader_full = DataLoader(ValData(), batch_size=val_batch_size, shuffle=False, num_workers=12)
# val_data_loader = val_data_loader_full

from datas.GoPro_new import get_loader
data_root = '/data/haodongli/Dataset_all/GOPRO'
test_batch_size = 1
num_workers = 8
val_data_loader = get_loader(data_root, test_batch_size, shuffle=False, num_workers=num_workers,
                         test_mode=True)

oup_psnr = []
oup_ssim = []
oup_lpips = 0

net.eval()

img_out_dir = '/data/haodongli/GDConvNet-main/out/gopro/'
# img_out_dir = '/data/haodongli/GDConvNet-main/out/vimeo90K7/gdc_ori_fast/'
# img_out_dir = '/data/haodongli/GDConvNet-main/out/vimeo90K7/gdc_ori_medium/'
# img_out_dir = '/data/haodongli/GDConvNet-main/out/vimeo90K7/gdc_ori_slow/'
if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)

transform = transforms.Compose([transforms.ToTensor()])
time_taken = []

mean = [0.5, 0.5, 0.5]
std  = [1, 1, 1]
revmean = [-x for x in mean]
revstd = [1.0 / x for x in std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])
revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])
revtrans1 = TF.Compose([revnormalize1, revnormalize2])

print("{}".format(len(val_data_loader)))

for batch_id, (images, gt_image)  in enumerate(val_data_loader):
    with torch.no_grad():
        img1, img2, img4, img5 = [img_.cuda() for img_ in images]
        gt = [g_.cuda() for g_ in gt_image]

        # img1, img2, img4, img5, gt = val_data
        # img1 = img1.to(device)
        # img2 = img2.to(device)
        # img4 = img4.to(device)
        # img5 = img5.to(device)
        # gt = gt.to(device)

        start_time = time.time()
        oup = net(img1, img2, img4, img5)
        time_taken.append(time.time() - start_time)

        gt = torch.cat(gt)

        # Calculate average PSNR
        oup_psnr.extend(to_psnr(oup, gt))
        # Calculate average SSIM
        oup_ssim.extend(pytorch_ssim.ssim(oup + 0.5, gt + 0.5, size_average=False).cpu().numpy())

        #print(to_psnr(oup, gt))
        imwrite(revtrans1(oup[0]), img_out_dir + '/' + str(batch_id) + '.png', range=(0, 1))
        #imwrite(revtrans1(gt), img_out_dir + '/' + str(batch_id) + '.0' + '.png', range=(0, 1))

        # imwrite(oup[0], img_out_dir + '/' + str(batch_id) + '.png', range=(0, 1))
        # imwrite(gt, img_out_dir + '/' + str(batch_id) + '.0' + '.png', range=(0, 1))

        ref = transform(Image.open(img_out_dir + '/' + str(batch_id) + '.png')).numpy()

        lps = lpips(gt.cuda(), torch.tensor(ref).unsqueeze(0).cuda(), net_type='squeeze')
        print('idx: %d, psnr: %f, ssim: %f, lpips: %f' % (batch_id, oup_psnr[-1], oup_ssim[-1], lps.item()))
        #print('idx: %d, psnr: %f, ssim: %f' % (batch_id, oup_psnr[-1], oup_ssim[-1]))
        oup_lpips += lps.item()
        torch.cuda.empty_cache()

print('successfully test {} images'.format(len(oup_psnr)))
#print('oup_PSNR:{0:.2f}, oup_SSIM:{1:.4f}'.format(sum(oup_psnr)/len(oup_psnr), sum(oup_ssim)/len(oup_ssim)))
msg = '\n{:<15s}{:<20.16f}{:<23.16f}{:<23.16f}'.format('Average: ', sum(oup_psnr)/len(oup_psnr), sum(oup_ssim)/len(oup_ssim), oup_lpips/len(oup_psnr))
print(msg)
print("Time , " , sum(time_taken)/len(time_taken))
print("All Time , " , time.time() - all_start_time)

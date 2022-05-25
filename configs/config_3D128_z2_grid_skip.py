learning_rate = 1e-3
num_epochs = 16
img_save_path = "./results/"
lr_schular = [1e-3, 5e-4, 1e-4, 2e-5, 1e-6]
training_schedule = [4, 8, 10, 12, 16]
device_id = [0,1]                              #

crop_height, crop_width = (256, 256)
train_batch_size = 2
val_batch_size = 8

mean = [0.5, 0.5, 0.5]
std  = [1, 1, 1]

train_data_dir = '/data/haodongli/vimeo_septuplet/'   # 数据集
val_data_dir = '/data/haodongli/vimeo_septuplet/'
from datas.val_data_vimeo import ValData

save_freq = 1

mode = 'poly'
delta = 0.5

if mode == 'poly':
    model_save_path = "./modeldict/poly_3D128_z2_grid_skip/"               # 走这里
elif mode == '1axis':
    model_save_path = "./modeldict/inverse_1axis_same/"
elif mode == '3axis':
    model_save_path = "./modeldict/inverse_3axis_same/"


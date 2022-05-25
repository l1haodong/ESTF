# ESTF
This is the official repository of ESTF (Enhanced Spatial-Temporal Freedom for Video Frame Interpolation).

Last Update: 20220525 -  The code of the proposed method has been submitted. 

# Prerequisites
The following pakages are required to run the code:
* Python == 3.7.6
* Pytorch == 1.2.0
* Torchvision == 0.4.0
* Pillow == 7.1.2
* Numpy == 1.18.1

# Datasets
Vimeo_septuplet is used as our training and testing dataset. Please download and unzip it somewhere on your device. Then change the training and testing directory in ./configs/config_3D_z2_grid_skip.py.

# Testing
We have released the model weight in ./modeldict/, you can directly use it to do the evaluation. Using the command to start the testing process.

```
$ python3 eval_myself.py 
```

# Training
You can also choose to retrain the model, just use the command to start the training process.
```
$ python3 train_full_model_3D_z2_grid_skip.py
```

# License
The source codes including the checkpoint can be freely used for research and education only. Any commercial use should get formal permission first.



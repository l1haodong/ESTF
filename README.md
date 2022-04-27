# ESTF
This is the official repository of ESTF (Enhanced Spatial-Temporal Freedom for Video Frame Interpolation).

Last Update: 20220427 -  The code of our proposed method is ready and will be available to public as soon as the manuscript is accepted. 

# Prerequisites
* Python == 3.7.6
* Pytorch == 1.2.0
* Torchvision == 0.4.0
* Pillow == 7.1.2
* Numpy == 1.18.1

# Datasets
Vimeo_septuplet is used as our training and testing dataset. Please download and unzip it somewhere on your device. Then change the training and testing directory in ./configs/config.py.

# Testing
We have released the model weight in ./modeldict/, you can directly use it to do the evaluation. Using the command to start the testing process.

```
$ python3 eval.py 
```



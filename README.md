# segNet_tensorflow: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation using tensorflow.

The implementation for SegNet is based on the paper listed below:

http://arxiv.org/abs/1511.02680 Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015.

## Prerequisites
 - Tensorflow 1.15.0
 - Numpy
 - Scipy
 - Glob
 - Numpy

## Dataset : CamVid 
 - Trainig images: 367
 - Testing images: 100
 - Validation Images: 100
 - Resolution: 360 x 480

# Python files
 - main.py: contains the SegNet model
 - segnet.py: contains the required functions for the loss, initialization and predictions function

# Results
| Optimizers | sky | Building | Pole | Road | Pavement | Tree | Sign | Fence | Car | Pedestrian | Bicycle | GA | mIoU |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Adam+MFB  | 94.249 | 68.725 | 18.690 | 67.270 | 85.460 | 80.800 | 74.480 | 0.0062 | 90.62 | 20.866 | 15.430 | 71.249 | 38.069 | 
| Adam | 77.710 | 68.500 | 0.9844 | 89.138 | 46.100 | 85.190 | 2.50 | 0.390 | 21.11 | 0.035 | 1.790 | 68.38 | 25 |
| SGD+MFB | 73.750 | 58.238 | 11.090 | 69.270 | 69.780 | 85.930 | 74.110 | 0.0019 | 87.45 | 31.75 | 27.28 | 65.91 | 37.29
| SGD | 89.97 | 66.09 | 3.13 | 67.18 | 75.78 | 64.8 | 45.419 | 0.000 | 88 | 16.880 | 39.10 | 66.159 | 28.68 |



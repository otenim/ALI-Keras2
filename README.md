# ALI-Keras2

## Overview

This repository provides an implementation of [Adversarially Learned Inference](https://arxiv.org/pdf/1606.00704) using a deeplearning library, Keras2.

## Dependency

* Python==3.5.\*, 3.6.\*
* Keras==2.0.\*
* numpy==1.12.\*, 1.13.\*
* Pillow==4.\*

All dependent libraries can be installed with `pip` command.  
**Note: We only tested our scripts using Keras with tensorflow backend.**  

## Run training scripts

You can run our training scripts with the following command.  
`$ python train.py [--epochs] [--batch_size] [--lr] [--beta_1] [--beta_2] [--snap_freq] [--result]`

* `--epochs`: training epochs.
* `--batch_size`: batch size during the training phase.
* `--lr`: learning rate (we use Adam as the optimizer).
* `--beta_1`: beta\_1 parameter of Adam.
* `--beta_2`: beta\_2 parameter of Adam.
* `--snap_freq`: save generaters' weights at each (snap\_freq) epochs.
* `--result`: a path to the directory where training results (generater's weights and generated images) are to be saved.

ex) `$ python train.py --epochs 500 --batch_size 100`

All the default values of the above arguments and hyper parameters of the network architecture are basically the same values which are used in the original paper.  

## Experiment results

We used Titan X (pascal architecture) X 4 for all the experiments.

### cifar10

* 50,000 training images.
* 10,000 test images are used to generate or reconstruct images.
* All the images' shapes are unified into (h, w, c) = (32, 32, 3).
* The number of classes is 10 (5,000 images for each class).

**epochs875(generate)**

![result1](https://i.imgur.com/VDrzbFi.png)  

**epochs875(reconstruct)**  

![result2](https://i.imgur.com/JILYa9h.png)  

**epochs3800(generate)**  

![result3](https://i.imgur.com/CbmATUT.png)  

**epochs3800(reconstruct)**  

![result4](https://i.imgur.com/b46SfPl.png)  

**epochs6000(generate)**  

![result5](https://i.imgur.com/kSDaHWk.png)  

**epochs6000(reconstruct)**  

![result6](https://i.imgur.com/bjRhI0s.png)


# ALI-Keras2

## Overview

This repository provides an implementation of [Adversarially Learned Inference](https://arxiv.org/pdf/1606.00704) using a deeplearning library, Keras2.

## Dependency

* Python==3.5.\*, 3.6.\*
* Keras==2.0.\*
* numpy==1.12.\*, 1.13.\*
* Pillow==4.\*

All dependent libraries above can be installed with `pip` command.

## Run training scripts

You can run our training scripts with the following command.  
`$ python train.py [--epochs] [--batch_size] [--lr] [--beta_1] [--beta_2] [--snap_freq] [--result_root]`

* `--epochs`: training epochs
* `--batch_size`: mini-batchsize while training phase
* `--lr`: learning rate. (NOTE: We use Adam as an optimizer)
* `--beta_1`: beta1 parameter of Adam
* `--beta_2`: beta2 parameter of Adam
* `--snap_freq`: save generaters' weights at each (snap_freq) epochs
* `--result_root`: a path to the directory which saves training results(generater's weights and generated images)

All initial values of above arguments and hyper parameters of network architectures are basically set with reference to ALI's paper.

ex) `$ python train.py --epochs 500 --batch_size 100`

## Experiment results

We use TitanX(pascal architecture)X4 for all experiments.

### cifar10

* 50,000 training images
* all images' size are unified into (h,w,c) = (32,32,3)
* the number of classes is 10(5,000 images for each class)  

**epochs315**

![result1](https://i.imgur.com/16buX2d.png)  

**epochs875**

![result2](https://i.imgur.com/VDrzbFi.png)  

We stoppped training at this epoch not because over fitting or kind of that had occured. So, there is a possibility that higher quality images may be generated with more training epochs or network tunings, I think.

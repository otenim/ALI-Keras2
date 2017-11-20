# ALI-Keras2

## Overview

This repository provides an implementation of [Adversarially Learned Inference](https://arxiv.org/pdf/1606.00704) using a deeplearning library, Keras2.

## Dependency

* Python==3.5.\*, 3.6.\*
* Keras==2.0.\*
* numpy==1.12.\*, 1.13.\*
* Pillow==4.\*

All dependent libraries above can be installed with `pip` command.  
**Note: We only tested our scripts using Keras with tensorflow batckend.**  

## Run training scripts

You can run our training scripts with the following command.  
`$ python train.py [--epochs] [--batch_size] [--lr] [--beta_1] [--beta_2] [--snap_freq] [--result]`

* `--epochs`: training epochs
* `--batch_size`: mini-batchsize while training phase
* `--lr`: learning rate. (NOTE: We use Adam as an optimizer)
* `--beta_1`: beta1 parameter of Adam
* `--beta_2`: beta2 parameter of Adam
* `--snap_freq`: save generaters' weights at each (snap\_freq) epochs
* `--result`: a path to the directory which saves training results(generater's weights and generated images)

All initial values of above arguments and hyper parameters of network architectures are basically set with reference to ALI's paper.

ex) `$ python train.py --epochs 500 --batch_size 100`

## Experiment results

We use TitanX(pascal architecture)X4 for all experiments.

### cifar10

* 50,000 training images
* 10,000 test images are used to generate or reconstruct images.
* all images' size are unified into (h,w,c) = (32,32,3)
* the number of classes is 10(5,000 images for each class)  

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

We stoppped training at this epoch not because over fitting or kind of that had occured. So, there is a possibility that higher quality images may be generated with more training epochs or network tunings, I think.

# Assignment
## Part 1
1. Write your own UNet from scratch 
2. Train 4 times: 
    - MP+Tr+CE 
    - MP+Tr+Dice Loss 
    - StrConv+Tr+CE 
    - StrConv+Ups+Dice Loss 

## Introduction
The goal of this assignment is to implement Unet model from scratch for segmentation task. Train it
with different type of losses and layers.

## MP+Tr+CE

### Train logs
    Epoch 1
    Train: Loss=0.5915 Batch_id=57: 100%|██████████| 58/58 [00:46<00:00,  1.24it/s]
    Test set: Average loss: 1.1637
    Epoch 2
    Train: Loss=0.5212 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.9668
    Epoch 3
    Train: Loss=0.4703 Batch_id=57: 100%|██████████| 58/58 [00:46<00:00,  1.24it/s]
    Test set: Average loss: 0.7513
    Epoch 4
    Train: Loss=0.4590 Batch_id=57: 100%|██████████| 58/58 [00:46<00:00,  1.23it/s]
    Test set: Average loss: 0.7183
    ...
    Epoch 21
    Train: Loss=0.2160 Batch_id=57: 100%|██████████| 58/58 [00:46<00:00,  1.24it/s]
    Test set: Average loss: 0.5025
    Epoch 22
    Train: Loss=0.1932 Batch_id=57: 100%|██████████| 58/58 [00:46<00:00,  1.24it/s]
    Test set: Average loss: 1.1327
    Epoch 23
    Train: Loss=0.1688 Batch_id=57: 100%|██████████| 58/58 [00:46<00:00,  1.23it/s]
    Test set: Average loss: 0.7333
    Epoch 24
    Train: Loss=0.1792 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.7884
    Epoch 25
    Train: Loss=0.1613 Batch_id=57: 100%|██████████| 58/58 [00:46<00:00,  1.24it/s]
    Test set: Average loss: 0.5817

### Performance Graph
![MP+Tr+CE metrics](./images/MP_Tr_CE_metrics.png)

### Results
![MP+Tr+CE results](./images/MP_Tr_CE_results.png)



## MP+Tr+Dice Loss

### Train logs
    CUDA Available? True
   Epoch 1
    Train: Loss=0.1484 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.2467
    Epoch 2
    Train: Loss=0.1479 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.2779
    Epoch 3
    Train: Loss=0.1404 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.2347
    Epoch 4
    Train: Loss=0.1140 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.2922
    Epoch 5
    Train: Loss=0.0940 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.2549
    ...
    EEpoch 21
    Train: Loss=0.0602 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.0989
    Epoch 22
    Train: Loss=0.0495 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.1216
    Epoch 23
    Train: Loss=0.0561 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.0917
    Epoch 24
    Train: Loss=0.0480 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.23it/s]
    Test set: Average loss: 0.0814
    Epoch 25
    Train: Loss=0.0484 Batch_id=57: 100%|██████████| 58/58 [00:47<00:00,  1.22it/s]
    Test set: Average loss: 0.0870

### Performance Graph
![MP+Tr+Dice Loss metrics](./images/MP_Tr_Dice_Loss_metrics.png)

### Results
![MP+Tr+Dice Loss results](./images/MP_Tr_Dice_Loss_results.png)



## StrConv+Tr+CE
>

### Train logs
    CUDA Available? True
    Epoch 1
    Train: Loss=0.6212 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 3.3839
    Epoch 2
    Train: Loss=0.6365 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.5912
    Epoch 3
    Train: Loss=0.5591 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.5569
    Epoch 4
    Train: Loss=0.4087 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.5307
    Epoch 5
    Train: Loss=0.5254 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.5210
    ...
    Epoch 21
    Train: Loss=0.2764 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.3245
    Epoch 22
    Train: Loss=0.3216 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.3163
    Epoch 23
    Train: Loss=0.3235 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.3246
    Epoch 24
    Train: Loss=0.2630 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.3082
    Epoch 25
    Train: Loss=0.2607 Batch_id=57: 100%|██████████| 58/58 [00:48<00:00,  1.20it/s]
    Test set: Average loss: 0.3117

### Performance Graph
![StrConv+Tr+CE metrics](./images/StrConv_Tr_CE_metrics.png)

### Results
![StrConv+Tr+CE results](./images/StrConv_Tr_CE_results.png)



## StrConv+Ups+Dice Loss


### Train logs
    CUDA Available? True
    Epoch 1
    Train: Loss=0.1853 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.2845
    Epoch 2
    Train: Loss=0.1176 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.1839
    Epoch 3
    Train: Loss=0.1280 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.1331
    Epoch 4
    Train: Loss=0.1377 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.10it/s]
    Test set: Average loss: 0.1281
    Epoch 5
    Train: Loss=0.1250 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.1096
    ...
    Epoch 21
    Train: Loss=0.0667 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.0688
    Epoch 22
    Train: Loss=0.0645 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.0707
    Epoch 23
    Train: Loss=0.0588 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.0659
    Epoch 24
    Train: Loss=0.0537 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.0673
    Epoch 25
    Train: Loss=0.0655 Batch_id=57: 100%|██████████| 58/58 [00:52<00:00,  1.11it/s]
    Test set: Average loss: 0.0656

### Performance Graph
![StrConv+Ups+Dice Loss metrics](./images/StrConv_Ups_Dice_Loss_metrics.png)

### Results
![StrConv+Ups+Dice Loss results](./images/StrConv_Ups_Dice_Loss_results.png)



## Part 2

1. Design a varition of a VAE to take image and it's label as input
2. Train 2 times:
    - Train on MNIST data
    - Train on CIFAR10 data
3. Generate 25 images from each model

## Introduction
The goal of this assignment is to implement VAE model from scratch for image generation task. Train it
with different public data.

## MNIST data

### train logs
    Epoch 29: 100% 1500/1500 [01:26<00:00, 17.27it/s, loss=-949, v_num=0]

### Results
![MNIST results](./images/mnist_results.png)



## CIFAR10 data


### train logs
    Epoch 29: 100% 1250/1250 [01:30<00:00, 13.80it/s, loss=-2.93e+03, v_num=1]

### Results
![CIFAR10 results](./images/CIFAR10_results.png)


## Acknowledgments
This model is trained using repo listed below
* [UNet](https://github.com/AkashDataScience/unet_pytorch)
* [VAE](https://github.com/AkashDataScience/vae_pytorch)
# How to use LRfinder and Onclycle policy
## Code Description
The code provided here implements of LRfinder and Once cycle police Convolutional Neural Network (CNN) using PyTorch for image classification tasks on the CIFAR-10 dataset. The architecture of the model consists of several convolutional layers mix of normal convo  layer as well as Resnet-18 model. The dataset is split into training and testing sets, with data augmentation techniques applied during training to improve model generalization.


## Findings
The code includes training and testing procedures to train the model and evaluate its performance. It tracks and plots the training and test losses, as well as the training and test accuracies over epochs. Additionally, it visualizes misclassified images along with their true and predicted labels to analyze the model's behavior.
- We use LRfinder to find max LR required to reach heighest traing and test accurecy
- One Cycle pollicy is quite usefull when we want to reach high accurecy while using less ammount of resouces and with less EPOCHs

## Graphs
Several graphs are plotted to visualize the training and testing process:

1. Training Loss: !(image.png)
2. Test Loss: !(image-1.png)
3. Training Accuracy: !(image-2.png)
4. Test Accuracy: !(image-3.png)


## Collection of Misclassified Images
The code generates a collection of misclassified images along with their actual and predicted labels.
(image-4.png)

## Training Accuracy
The training accuracy: 

## Test Accuracy
The test accuracy: 91.79
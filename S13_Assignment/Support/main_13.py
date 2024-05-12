import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#import model
import matplotlib.pyplot as plt
import numpy as np
#%cd ERA-V2-Support
from Support.model_13 import ResNet18
import Support.dataset
import Support.train
import Support.utils
from pytorch_lightning import Trainer 



SEED = 1


cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)


torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

train_data = Support.dataset.train()

test_data = Support.dataset.test()

if cuda:
    batch_size = 128
    shuffle = True
    num_workers = 2
    pin_memory = True
    train_loader = Support.dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,train_data)
    test_loader = Support.dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,test_data)
else:
    batch_size =64
    shuffle = True
    num_workers =1
    pin_memory = True
    train_loader = Support.dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,train_data)
    test_loader = Support.dataset.get_train_loader(batch_size,shuffle,num_workers,pin_memory,test_data)

classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
Support.utils.plot_image(train_loader)

device = Support.utils.get_device()
model = ResNet18().to(device)
EPOCHS = 25

trainer = Trainer(
    accelerator = device,
    max_epochs = EPOCHS,
    enable_progress_bar = True
    #progress_bar_refresh_rate = 20,
)

trainer.fit(model, train_loader, test_loader)

trainer.test(model, test_loader)

#train.plot_loss_accuracy(train_losses,train_acc,test_losses,test_acc)

misclassified_data = Support.utils.get_misclassified_data(model, device, test_loader)

inv_normalize = transforms.Normalize(
    mean=[-0.50/0.4914, -0.50/0.4822, -0.50/0.4465],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)

Support.utils.display_cifar_misclassified_data(misclassified_data, classes, inv_normalize, number_of_samples=10)

target_layers = [model.layer4[-1]]
# targets = [ClassifierOutputTarget(7)]
targets = None

Support.utils.display_gradcam_output(misclassified_data, classes, inv_normalize, model, target_layers, targets, number_of_samples=10, transparency=0.70)

torch.save(model.state_dict(), "model.pth")
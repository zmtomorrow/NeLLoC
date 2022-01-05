import os
import torch
from torch import optim
from torch.utils import data
import torch.nn as nn
from model import *
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import *



transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()

])

trainset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=True, download=False, transform=transform_train)
train = data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=3)

testset = torchvision.datasets.CIFAR10(root='../data/cifar10', train=False, download=False, transform=transform_train)
test = data.DataLoader(testset, batch_size=1000, shuffle=True, num_workers=3)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = LocalPixelCNN(res_num=0, in_kernel = 7,  in_channels=3, channels=256, out_channels=100).to(device)


optimizer = optim.Adam(net.parameters(), lr=3e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99995)

criterion  = lambda real, fake : discretized_mix_logistic_uniform(real, fake, alpha=0.0001)

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5



for e in range(1001):
    print('epoch',e)
    net.train()
    for images, labels in train:
        images = rescaling(images).to(device)

        optimizer.zero_grad()

        output = net(images)
        loss = criterion(images, output)
        loss.backward()
        optimizer.step()    
    scheduler.step()

    
    
    with torch.no_grad():
        net.eval()
        bpd_cifar_sum=0.

        for i, (images, labels) in enumerate(test):
            images = rescaling(images).to(device)
            output = net(images)
            loss = criterion(images, output).item()
            bpd_cifar_sum+=loss/(np.log(2.)*(1000*32*32*3))
        bpd_cifar=bpd_cifar_sum/10
        print('bpd_cifar',bpd_cifar)

    
    save_path='./model_save/'
    torch.save(net.state_dict(),    save_path+'rs0_cifar_ks7.pt')

        

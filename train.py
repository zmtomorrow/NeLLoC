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

def discretized_mix_logistic_uniform(x, l, alpha=0.0001):
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]
    nr_mix = int(ls[-1] / 10) 
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
   
    coeffs = F.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
                * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
                coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = F.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = F.sigmoid(min_in)
    cdf_plus=torch.where(x > 0.999, torch.tensor(1.0).to(x.device),cdf_plus)
    cdf_min=torch.where(x <- 0.999, torch.tensor(0.0).to(x.device),cdf_min)
    uniform_cdf_min = ((x+1.)/2*255)/256.
    uniform_cdf_plus = ((x+1.)/2*255+1)/256.
    pi=torch.softmax(logit_probs,-1).unsqueeze(-2).repeat(1,1,1,3,1)
    mix_cdf_plus=((1-alpha)*pi*cdf_plus+(alpha/nr_mix)*uniform_cdf_plus).sum(-1)
    mix_cdf_min=((1-alpha)*pi*cdf_min+(alpha/nr_mix)*uniform_cdf_min).sum(-1)
    log_probs =torch.log(mix_cdf_plus-mix_cdf_min)
    return -log_probs.sum()

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

        

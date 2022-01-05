'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''

from torchvision import datasets, transforms
import configparser
import os
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
from PIL import *
import os
import pickle

normalize=lambda x: x/255.




def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict



class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = []
        for (dirpath, dirnames, filenames) in os.walk(self.main_dir):
            self.all_imgs.extend([os.path.join(dirpath, file) for file in filenames])
        self.all_imgs=sorted(self.all_imgs)
        

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = self.all_imgs[idx]
#         print(img_loc)
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image



def parse_value_from_string(val_str):
    if(is_int(val_str)):
        val = int(val_str)
    elif(is_float(val_str)):
        val = float(val_str)
    elif(is_list(val_str)):
        val = parse_list(val_str)
    elif(is_bool(val_str)):
        val = parse_bool(val_str)
    else:
        val = val_str
    return val

def is_int(val_str):
    start_digit = 0
    if(val_str[0] =='-'):
        start_digit = 1
    flag = True
    for i in range(start_digit, len(val_str)):
        if(str(val_str[i]) < '0' or str(val_str[i]) > '9'):
            flag = False
            break
    return flag

def is_float(val_str):
    flag = False
    if('.' in val_str and len(val_str.split('.'))==2):
        if(is_int(val_str.split('.')[0]) and is_int(val_str.split('.')[1])):
            flag = True
        else:
            flag = False
    elif('e' in val_str and len(val_str.split('e'))==2):
        if(is_int(val_str.split('e')[0]) and is_int(val_str.split('e')[1])):
            flag = True
        else:
            flag = False       
    else:
        flag = False
    return flag 

def is_bool(var_str):
    if( var_str=='True' or var_str == 'true' or var_str =='False' or var_str=='false'):
        return True
    else:
        return False
    
def parse_bool(var_str):
    if(var_str=='True' or var_str == 'true' ):
        return True
    else:
        return False
     
def is_list(val_str):
    if(val_str[0] == '[' and val_str[-1] == ']'):
        return True
    else:
        return False
    
def parse_list(val_str):
    sub_str = val_str[1:-1]
    splits = sub_str.split(',')
    output = []
    for item in splits:
        item = item.strip()
        if(is_int(item)):
            output.append(int(item))
        elif(is_float(item)):
            output.append(float(item))
        elif(is_bool(item)):
            output.append(parse_bool(item))
        else:
            output.append(item)
    return output


    
def show_many(image,number_sqrt,dim=32, channels=3):
    if image.size(1)==3:
        image=image.permute(0,2,3,1)
    canvas_recon = np.empty((dim * number_sqrt, dim * number_sqrt, channels))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim,:] = \
            image[count]
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon)
    plt.show()
    
def grey_show_many(image,number_sqrt):
    
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

def color_sample(net, device,num_sqrt=10):
    sample = torch.Tensor(num_sqrt**2, 3, 32, 32).to(device)
    sample.fill_(0)
    with torch.no_grad():
    #Generating images pixel by pixel
        for i in range(32):
            for j in range(32):
                out = net(sample).view(-1,256,3,32,32)[:,:,:,i,j].permute(0,2,1).contiguous() 
                probs = F.softmax(out.view(-1,256), dim=-1).data
                sample[:,:,i,j] = (torch.multinomial(probs, 1).float() / 255.0).view(-1,3)

        #Saving images row wise
        show_many(sample.detach().cpu(),num_sqrt)
        
        
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


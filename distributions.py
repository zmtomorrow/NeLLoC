import torch
import numpy as np

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

def discretized_mix_logistic_cdftable(means, log_scales,pi, alpha=0.0001):
    x=rescaling(torch.arange(0,256)/255.).view(256,1).repeat(1,10).to(means.device)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    cdf_plus=torch.where(x > 0.999, torch.tensor(1.0).to(x.device),cdf_plus)
    cdf_min=torch.where(x <- 0.999, torch.tensor(0.0).to(x.device),cdf_min)

    uniform_cdf_min = ((x+1.)/2*255)/256.
    uniform_cdf_plus = ((x+1.)/2*255+1)/256.

    mix_cdf_plus=((1-alpha)*pi*cdf_plus+(alpha/10)*uniform_cdf_plus).sum(-1)
    mix_cdf_min=((1-alpha)*pi*cdf_min+(alpha/10)*uniform_cdf_min).sum(-1)
    return mix_cdf_plus,mix_cdf_min


def compute_stats(l):
    nr_mix=10
    pi=torch.softmax(l[:nr_mix],-1)
    l=l[nr_mix:].view(3,30)
    means=l[:,:nr_mix]
    log_scales = torch.clamp(l[:,nr_mix:2 * nr_mix], min=-7.)
    coeffs = torch.tanh(l[:,2 * nr_mix:3 * nr_mix])
    return means,coeffs,log_scales, pi 


def cdf_table_processing(cdf_plus,cdf_min,p_prec):
    p_total=1<<p_prec
    cdf_min=np.rint(cdf_min.numpy()* p_total)
    cdf_plus=np.rint(cdf_plus.numpy()* p_total)
    probs=cdf_plus-cdf_min
    probs[probs==0]=1
    probs[np.argmax(probs)]+=(p_total-np.sum(probs))
    return np.concatenate(([0],np.cumsum(probs)[:-1])),probs


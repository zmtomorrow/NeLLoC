from distributions import *
import torch
import numpy as np

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

def discretized_mix_logistic_cdftable(means, log_scales,pi, alpha=0.0001):
    bs=means.size(0)
    pi=pi.unsqueeze(1)
    x=rescaling(torch.arange(0,256)/255.).view(1,256,1).repeat(bs,1,10)
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
    bs=l.size(0)
    nr_mix=10
    pi=torch.softmax(l[:,:nr_mix],-1)
    l=l[:,nr_mix:].view(bs,3,30)
    means=l[:,:,:nr_mix]
    log_scales = torch.clamp(l[:,:,nr_mix:2 * nr_mix], min=-7.)
    coeffs = torch.tanh(l[:,:,2 * nr_mix:3 * nr_mix])
    return means,coeffs,log_scales, pi 

def get_mean_c1(means,mean_linear,x):
    return means+x.unsqueeze(-1)*mean_linear

def get_mean_c2(means,mean_linear,x):
    return means+torch.bmm(x.view(-1,1,2),mean_linear.view(-1,2,10)).view(-1,1,10)


def cdf_table_processing(cdf_plus,cdf_min,p_prec):
    p_total=np.asarray((1 << p_prec),dtype='uint32')
    bs=cdf_plus.size(0)
    cdf_min=np.rint(cdf_min.numpy()*  p_total).astype('uint32')
    cdf_plus=np.rint(cdf_plus.numpy()* p_total).astype('uint32')
    probs=cdf_plus-cdf_min
    probs[probs==0]=1
    argmax_index=np.argmax(probs,axis=1).reshape(-1,1)
    diff=p_total-np.sum(probs,-1,keepdims=True)
    value=diff+np.take_along_axis(probs, argmax_index.reshape(-1,1), axis=-1)
    np.put_along_axis(probs, argmax_index,value , axis=-1)
    return np.concatenate((np.zeros((bs,1),dtype='uint32'),np.cumsum(probs[:,:-1],axis=-1,dtype='uint32')),1),probs


s_prec = 64
t_prec = 32
t_mask = (1 << t_prec) - 1
s_min  = 1 << s_prec - t_prec
s_max  = 1 << s_prec
s_prec_u,t_prec_u=np.uint8(s_prec),np.uint8(t_prec)


def get_length(s,t_stack):
    return  ((len(t_stack))*t_prec+sum(len(bin(i)) for i in s))/(32*32*3)

def cpu_ans_compression(model,img,h=32,w=32,k=7,p_prec=20):
    p_prec_u=np.uint8(p_prec)

    c_list=[]
    p_list=[]
    rf=int(k/2)

    with torch.no_grad():
        for i in range(0,h):
            for j in range(0,w):
                
                up=max(0,i-rf)
                left=max(0,j-rf)
                down=i+1
                right=j+1+int(i>0)*rf
                m,n=min(rf,i),min(rf,j)
                patch_int=img[:,:,up:down,left:right]
                patch=rescaling(patch_int/255.)
                model_output=model(patch)
                means,coeffs,log_scales, pi=compute_stats(model_output[:,:,m,n].view(img.size(0),-1))
                c_0=rescaling(img[:,0:1,i,j]/255.)
                c_1=rescaling(img[:,0:2,i,j]/255.)

                
                for c in range(0,3):  
                    if c==0:
                        mean=means[:,0:1, :]
                    elif c==1:
                        mean=get_mean_c1(means[:,1:2, :], coeffs[:,0:1, :],c_0)
                    else:
                        mean=get_mean_c2(means[:,2:3, :], coeffs[:,1:3, :],c_1)

                    cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                    

                    c_list.append(np.take_along_axis(cdf_min_table,patch_int[:,c,m,n].numpy().reshape(-1,1),axis=-1).reshape(-1))
                    p_list.append(np.take_along_axis(probs_table,patch_int[:,c,m,n].numpy().reshape(-1,1),axis=-1).reshape(-1))                       
    s, t_stack = np.asarray([s_min]*img.size(0),dtype='uint64'), []
    for i in np.arange(len(c_list)-1,-1,-1):
        c_min,p=c_list[i],p_list[i]
        pos= s>>(s_prec - p_prec) >= p
        while True in pos:
            t_stack.extend(np.uint32(s[pos]))
            s[pos]>>= t_prec_u
            pos= s>>(s_prec_u - p_prec_u) >= p
        s = (s//p << p_prec_u) + s%p + c_min

    
    return s,t_stack



def cpu_ans_decompression(model,s,t_stack,bs,h=32,w=32,k=7,p_prec=20):
    p_prec_u=np.uint8(p_prec)

    with torch.no_grad():
#         device=next(model.parameters()).device
        rf=int(k/2)
        decode_img=torch.zeros([bs,3,h,w])
        for i in range(0,h):
            for j in range(0,w):
                up=max(0,i-rf)
                left=max(0,j-rf)
                down=i+1
                right=j+1+int(i>0)*rf
                patch=rescaling(decode_img[:,:,up:down,left:right]/255.)
                m,n=min(rf,i),min(rf,j)
                model_output=model(patch)
                means,coeffs,log_scales, pi=compute_stats(model_output[:,:,m,n].view(bs,-1))
                for c in range(0,3):
                    if c==0:
                        mean=means[:,0:1, :]
                    elif c==1:
                        mean=get_mean_c1(means[:,1:2, :], coeffs[:,0:1, :],rescaling(decode_img[:,0:1,i,j]/255.))
                    else:
                        mean=get_mean_c2(means[:,2:3, :], coeffs[:,1:3, :],rescaling(decode_img[:,0:2,i,j]/255.))
                    
                    cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[:,c:c+1],pi),p_prec)
                    s_bar = s & np.uint64(((1 << p_prec) - 1))
                    pt=np.asarray([np.searchsorted(cdf_min_table[i], s_bar[i], side='right', sorter=None)-1 for i in range(0,bs)])

                    decode_img[:,c,i,j]=torch.tensor(pt)
                    
                    patch[:,c,m,n]=torch.tensor(pt/255. )
                    
                    c_min=np.take_along_axis(cdf_min_table,pt.reshape(-1,1),axis=-1).reshape(-1)
                    p=np.take_along_axis(probs_table,pt.reshape(-1,1),axis=-1).reshape(-1)
                    s = p * (s >> np.uint8(p_prec)) + s_bar - c_min
                    # Renormalize
                    pos= s < s_min
                    while True in pos:
                        t_top=t_stack[-sum(pos):]
                        del t_stack[-sum(pos):]
                        s[pos] = (s[pos] << t_prec_u) + t_top
                        pos= s < s_min
#                     for yo in s:
#                         assert s_min <= yo < s_max
        return decode_img




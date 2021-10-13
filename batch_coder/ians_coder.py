import torch
import numpy as np
from batch_coder.distributions import *


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




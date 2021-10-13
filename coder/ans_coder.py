from coder.distributions import *
import numpy as np


class ANSStack(object):

    def __init__(self, s_prec , t_prec, p_prec):
        self.s_prec=s_prec
        self.t_prec=t_prec
        self.p_prec=p_prec
        self.t_mask = (1 << t_prec) - 1
        self.s_min=1 << s_prec - t_prec
        self.s_max=1 << s_prec
        self.s, self.t_stack= self.s_min, [] 

    def push(self,c_min,p):
        while self.s >= p << (self.s_prec - self.p_prec):
            self.t_stack.append(self.s & self.t_mask )
            self.s=self.s>> self.t_prec
        self.s = (self.s//p << self.p_prec) + self.s%p + c_min
        assert self.s_min <= self.s < self.s_max

    def pop(self):
        return self.s & ((1 << self.p_prec) - 1)

    def update(self,s_bar,c_min,p):
        self.s = p * (self.s >> self.p_prec) + s_bar - c_min
        while self.s < self.s_min:
            t_top=self.t_stack.pop()
            self.s = (self.s << self.t_prec) + t_top
        assert self.s_min <= self.s < self.s_max
        
    def get_length(self):
        return len(self.t_stack)*self.t_prec+len(bin(self.s))
        
        

def get_length(s,t_stack):
    return  ((len(t_stack))*16+len(bin(s)))/(32*32*3)

def cpu_ans_compression(model,img,h,w,k,p_prec=16):
    
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
                model_output=model(rescaling(patch_int/255.))
                means,coeffs,log_scales, pi=compute_stats(model_output[0,:,m,n].view(-1))
                c_0=rescaling(int(img[0,0,i,j])/255.)
                c_1=rescaling(int(img[0,1,i,j])/255.)
                for c in range(0,3):    
                    if c==0:
                        mean=means[0:1, :]
                    elif c==1:
                        mean=means[1:2, :] + coeffs[0:1, :]* c_0
                    else:
                        mean=means[2:3, :] + coeffs[1:2, :]* c_0 +coeffs[ 2:3, :] *  c_1
                    cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[c:c+1],pi),p_prec)

                    c_list.append(int(cdf_min_table[patch_int[0,c,m,n]]))
                    p_list.append(int(probs_table[patch_int[0,c,m,n]]))


    ans_stack=ANSStack(s_prec = 32,t_prec = 16, p_prec=p_prec)              
    for i in np.arange(len(c_list)-1,-1,-1):
        c_min,p=c_list[i],p_list[i]
        ans_stack.push(c_min,p)
    return ans_stack                



def cpu_ans_decompression(model,ans_stack,h,w,k,p_prec=16):
    with torch.no_grad():
        rf=int(k/2)

        decode_img=torch.zeros([1,3,h,w])
        for i in range(0,h):
            for j in range(0,w):
                up=max(0,i-rf)
                left=max(0,j-rf)
                down=i+1
                right=j+1+int(i>0)*rf
                patch=decode_img[:,:,up:down,left:right]
                m,n=min(rf,i),min(rf,j)
                model_output=model(rescaling(patch/255.))
                means,coeffs,log_scales, pi=compute_stats(model_output[0,:,m,n].view(-1))
                c_vector=[0,0,0]
                for c in range(0,3):
                    if c==0:
                        mean=means[0:1, :]
                    elif c==1:
                        mean=means[1:2, :] + coeffs[0:1, :]* c_vector[0]
                    else:
                        mean=means[2:3, :] + coeffs[1:2, :]* c_vector[0] +coeffs[2:3, :] *  c_vector[1]
                    cdf_min_table,probs_table= cdf_table_processing(*discretized_mix_logistic_cdftable(mean,log_scales[c:c+1],pi),p_prec)
                    s_bar = ans_stack.pop()
                    pt=np.searchsorted(cdf_min_table, s_bar, side='right', sorter=None)-1
                    decode_img[0,c,i,j]=pt
                    c_vector[c]=torch.tensor(rescaling(pt/255.))
#                     patch[0,c,m,n]=pt/255. 
                    c,p=int(cdf_min_table[pt]),int(probs_table[pt])
                    ans_stack.update(s_bar,c,p)
        return decode_img[0]
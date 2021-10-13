from model import *
from decimal import *
from coder.distributions import *
tensor2decimal= lambda x : Decimal(str(x.cpu().item()))



def bin_2_float(binary):
    prob = Decimal('0.0')
    cur_prob=Decimal('0.5')
         
    for i in binary:
        prob=prob+cur_prob* int(i)
        cur_prob*=Decimal('0.5')
    return prob


def range_2_bin(low, high):
    code = []
    prob = Decimal('0.0')
    cur_prob=Decimal('0.5')
    
    while(prob < low):
        acc_prob=prob+cur_prob
        if acc_prob > high:
            code.append(0)
        else:
            code.append(1)
            prob = acc_prob
        cur_prob*=Decimal('0.5')
    return code


def cpu_ac_compression(model,img,k):
    with torch.no_grad():
        model.eval()
        device=next(model.parameters()).device
        rf=int(k/2)
        size=img.size()
        total_size=size[1]*size[2]*size[3]
        old_low  = Decimal('0.0')
        old_high = Decimal('1.0')
        _range   = Decimal('1.0')
        for i in range(0,size[2] ):
            for j in range(0,size[3]):
                up=max(0,i-rf)
                left=max(0,j-rf)
                down=i+1
                right=j+1+int(i>0)*rf
                patch=rescaling(img[:,:,up:down,left:right]/255.)
                m,n=min(rf,i),min(rf,j)
                model_output=model(patch.to(device))

                means,coeffs,log_scales, pi=compute_stats(model_output[0,:,m,n].view(-1))
                c_0=rescaling(int(img[0,0,i,j])/255.)
                c_1=rescaling(int(img[0,1,i,j])/255.)
                for c in range(0,3):    
                    if c==0:
                        mean=means[0:1, :]
                    elif c==1:
                        mean=means[1:2, :] + coeffs[0:1, :]* c_0
                    else:
                        mean=means[2:3, :] + coeffs[1:2, :]* c_0 +coeffs[ 2:3, :] * c_1
                        

                    cdf_plus,cdf_min= discretized_mix_logistic_cdftable(mean,log_scales[c:c+1],pi)
                    
                    low  = old_low + _range * tensor2decimal(cdf_min[int(img[0,c,i,j])])
                    high = old_low + _range * tensor2decimal(cdf_plus[int(img[0,c,i,j])])
                    _range = high - low
                    old_low  = low
                    old_high = high
    code=range_2_bin(low,high)
    return code
    


    
def cpu_ac_decompression(model,code,h,w,k):
    model.eval()
    device=next(model.parameters()).device
    with torch.no_grad():
        prob = bin_2_float(code)
        low = Decimal(0.0)
        high = Decimal(1.0)
        _range = Decimal(1.0)
        rf=int(k/2)
        decode_img=torch.zeros([1,3,h,w])
        for i in range(0,h):
            for j in range(0,w):
                up=max(0,i-rf)
                left=max(0,j-rf)
                down=i+1
                right=j+1+int(i>0)*rf
                patch=decode_img[:,:,up:down,left:right].clone()
                m,n=min(rf,i),min(rf,j)
                model_output=model(rescaling(patch/255.).to(device))

                means,coeffs,log_scales, pi=compute_stats(model_output[0,:,m,n].view(-1))
                c_vector=[0,0,0]
                for c in range(0,3):    
                    if c==0:
                        mean=means[0:1, :]
                    elif c==1:
                        mean=means[1:2, :] + coeffs[0:1, :]* c_vector[0]
                    else:
                        mean=means[2:3, :] + coeffs[1:2, :]* c_vector[0] +coeffs[2:3, :] *  c_vector[1]
                        
                    cdf_plus,cdf_min= discretized_mix_logistic_cdftable(mean,log_scales[c:c+1],pi)
                    s=128
                    bl=0
                    br=256
                    for bs in range(0,9):
                        if  tensor2decimal(cdf_min[s])>prob:
                            br=s
                            s=int((s+bl)/2)
                        elif tensor2decimal(cdf_plus[s])<prob:
                            bl=s
                            s=int((s+br)/2)
                        else:
                            decode_img[0,c,i,j]=s
                            low=tensor2decimal(cdf_min[s])
                            high=tensor2decimal(cdf_plus[s])
                            c_vector[c]=torch.tensor(rescaling(s/255.))
                            _range=high-low
                            prob=(prob-low)/_range  
                            break
        return  decode_img[0]
    
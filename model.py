import torch.nn as nn
import torch
import torch.nn.functional as F



class MaskedCNN(nn.Conv2d):

    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = torch.zeros(1)
            self.mask[:,:,height//2+1:,:] = torch.zeros(1)
        else:
            self.mask[:,:,height//2,width//2+1:] = torch.zeros(1)
            self.mask[:,:,height//2+1:,:] = torch.zeros(1)


    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskedCNN, self).forward(x)
    
class PixelCNN_light(nn.Module):
    def __init__(self, in_kernel = 7,  in_channels=3, channels=100, out_channels=100, device=None):
        super(PixelCNN_light, self).__init__()
        self.channels = channels
        self.layers = {}
        self.device = device
        

        self.in_cnn=MaskedCNN('A',in_channels,channels, in_kernel, 1, in_kernel//2, bias=False)
        self.out_cnn1=nn.Conv2d(channels, channels, 1)
        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)

        
    def forward(self, x):
        x=F.leaky_relu(self.in_cnn(x))
        x=F.leaky_relu(self.out_cnn1(x))
        x=self.out_cnn2(x)
        return x


class LocalPixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al. 
    """
    def __init__(self, res_num=10, in_kernel = 7,  in_channels=3, channels=256, out_channels=256, device=None):
        super(LocalPixelCNN, self).__init__()
        self.channels = channels
        self.layers = {}
        self.device = device
        self.res_num=res_num
        

        self.in_cnn=MaskedCNN('A',in_channels,channels, in_kernel, 1, in_kernel//2, bias=False)
        self.activation=nn.ReLU()

        self.resnet_cnn11=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn3=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
        self.resnet_cnn12=torch.nn.ModuleList([MaskedCNN('B',channels,channels, 1, 1, 0) for i in range(0,res_num)])
 
        self.out_cnn1=nn.Conv2d(channels, channels, 1)
        self.out_cnn2=nn.Conv2d(channels, out_channels, 1)

        
    def forward(self, x):
        x=self.in_cnn(x)
        x=self.activation(x)

        for i in range(0, self.res_num):
            x_mid=self.resnet_cnn11[i](x)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn3[i](x_mid)
            x_mid=self.activation(x_mid)
            x_mid=self.resnet_cnn12[i](x_mid)
            x_mid=self.activation(x_mid)
            x=x+x_mid
        x=self.out_cnn1(x)
        x=self.activation(x)
        x=self.out_cnn2(x)
        return x
    
    
    

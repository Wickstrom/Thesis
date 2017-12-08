import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,n_in, n_out,l,activation):
        super(Encoder, self).__init__()
        
        self.l = l
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        
        self.block = nn.Sequential(
            *([nn.Conv2d(self.n_in, self.n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(self.n_out),
               self.activation,])
        )
        self.block_out = nn.Sequential(
            *([nn.Conv2d(self.n_out, self.n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(self.n_out),
               self.activation]*l)
        )
        self.pool = nn.MaxPool2d(2,2,return_indices=True)
    def forward(self,x):
        
        out = self.block(x)
        out = self.block_out(out)
        out, idx = self.pool(out)
        
        return out, idx

class Decoder(nn.Module):
    def __init__(self,n_in, n_out,l,activation):
        super(Decoder, self).__init__()
        
        self.l = l
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        
        self.block = nn.Sequential(
            *([nn.Conv2d(self.n_in, self.n_in, kernel_size=3, padding=1),
               nn.BatchNorm2d(self.n_in),
               self.activation,] * l)
        )
        self.block_out = nn.Sequential(
            *([nn.Conv2d(self.n_in, self.n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(self.n_out),
               self.activation])
        )
    
        self.unpool = nn.MaxUnpool2d(2,2)
    def forward(self,x,idx):

        out = self.unpool(x,idx)        
        out = self.block(out)
        out = self.block_out(out)
        
        return out
    
class SegNet(nn.Module):
    def __init__(self, num_classes,activation):
        super(SegNet, self).__init__()
        
        self.activation = activation
        self.num_classses = num_classes

        self.enc1 = Encoder(3,64,1,self.activation)
        self.enc2 = Encoder(64,128,1,self.activation)
        self.enc3 = Encoder(128,256,2,self.activation)
        self.enc4 = Encoder(256,512,2,self.activation)
        self.enc5 = Encoder(512,512,2,self.activation)
                
        self.dec5 = Decoder(512,512,2,self.activation)
        self.dec4 = Decoder(512,256,2,self.activation)
        self.dec3 = Decoder(256,128,2,self.activation)
        self.dec2 = Decoder(128,64,1,self.activation)
        self.dec1 = nn.Sequential(
            *([
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               self.activation,
               nn.Conv2d(64, self.num_classses, kernel_size=3, padding=1)])
            )
    
    def forward(self, x):
        enc1, idx1 = self.enc1(x)
        enc2, idx2 = self.enc2(enc1)
        enc3, idx3 = self.enc3(F.dropout2d(enc2))
        enc4, idx4 = self.enc4(F.dropout2d(enc3))
        enc5, idx5 = self.enc5(F.dropout2d(enc4))       
        
        dec5 = self.dec5(F.dropout2d(enc5),idx5)
        dec4 = self.dec4(F.dropout2d(dec5),idx4)
        dec3 = self.dec3(F.dropout2d(dec4),idx3)
        dec2 = self.dec2(dec3,idx2)
        dec1 = self.dec1(F.max_unpool2d(dec2,idx1,2,2))

        out = dec1.permute(1,0,2,3).contiguous()
        out = out.view(2,-1)
        out = out.permute(1,0)

        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################################################################
# Implementation of U-net from https://arxiv.org/abs/1505.04597.                                           #
# Note that this implementation assumes that input images have the property that                           #
# modulo(Input H/W,5) = 0. This is to handle size information during the upsampling. The                   #
# easiest way to ensure this is to either pad images and crop when calculating loss                        #
# or crop incoming images to size (224,224) for example.                                                   #
# Not the smoothest method to handle shape but i included this work as a part of data pre-processing.      #
############################################################################################################

class Encoder(nn.Module):                                                   
    def __init__(self, n_in,n_out,activation):
        super(Encoder, self).__init__()
        
        self.n_in = n_in                                                   # Number of features in.
        self.n_out = n_out                                                 # Number of features out.
        self.activation = activation
        
        self.block = nn.Sequential(                                        # Input encoder block.
            *([nn.Conv2d(self.n_in, self.n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(self.n_out),
               self.activation,])
        )
        self.block_out = nn.Sequential(                                    # Ouput encoder block. 
            *([nn.Conv2d(self.n_out, self.n_out, kernel_size=3, padding=1),
               nn.BatchNorm2d(self.n_out),
               self.activation])
        )
    def forward(self,x):                                                   # Forward pass for encoder.
        
        out = self.block(x)
        out = self.block_out(out)
        
        return(out)
    
class Decoder(nn.Module):
    def __init__(self, n_in,n_mid,n_out,activation):
        super(Decoder, self).__init__()
        
        self.n_in = n_in                                                  # Number of features in.
        self.n_mid = n_mid                                                # Number of features for center par of decoder.
        self.n_out = n_out                                                # Number of features out of decoder.
        self.activation = activation                                          
        
        self.block = nn.Sequential(                                       # Input decoder.
            *([nn.Conv2d(self.n_in, self.n_mid, kernel_size=3, padding=1),
               nn.BatchNorm2d(self.n_mid),
               self.activation,])
        )
        self.block_out = nn.Sequential(                                    # Output decoder. Consists of conv layer, BN, activation
            *([nn.Conv2d(self.n_mid, self.n_out, kernel_size=3, padding=1),# and transposed conv layer which upsamples by a factor
               nn.BatchNorm2d(self.n_out),                                 # of two.
               self.activation,
               nn.ConvTranspose2d(self.n_out,self.n_out,
                                  kernel_size=4,stride=2,padding=1,bias=False)])
        )
    def forward(self,x):                                                  # Forward pass for decoder
        
        out = self.block(x)
        out = self.block_out(out)
        
        return(out)

class Unet(nn.Module):
    def __init__(self, num_classes,activation):
        super(Unet, self).__init__()
        
        self.activation = activation
     
        self.enc1 = Encoder(3,64,self.activation)
        self.enc2 = Encoder(64,128,self.activation)
        self.enc3 = Encoder(128,256,self.activation)
        self.enc4 = Encoder(256,512,self.activation)
        
        self.center = nn.Sequential(                                       # Center block has dropout included as 
            *([nn.Dropout2d(),                                             # regularization, with a dropout-rate of 0.5.
               nn.Conv2d(512, 1024, kernel_size=3, padding=1),
               nn.BatchNorm2d(1024),
               self.activation,
               nn.Dropout2d(),
               nn.Conv2d(1024, 512, kernel_size=3, padding=1),
               nn.BatchNorm2d(512),
               self.activation,
               nn.ConvTranspose2d(512,512,
                                  kernel_size=4,stride=2,padding=1,bias=False)              
               ])
            )
    
        self.dec4 = Decoder(1024,512,256,self.activation)
        self.dec3 = Decoder(512,256,128,self.activation) 
        self.dec2 = Decoder(256,128,64,self.activation)    
        self.dec1 = nn.Sequential(                                         # Last part of decoder which outputs the number of  
            *([nn.Conv2d(128,64, kernel_size=3, padding=1),                # classes.
               nn.BatchNorm2d(64),
               self.activation,
               nn.Conv2d(64,64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               self.activation,
               nn.Conv2d(64,num_classes, kernel_size=3, padding=1)               
               ])
            )                
    
    def forward(self,x):                                                    
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1,kernel_size=2,stride=2))
        enc3 = self.enc3(F.max_pool2d(enc2,kernel_size=2,stride=2))
        enc4 = self.enc4(F.max_pool2d(enc3,kernel_size=2,stride=2))

        center = self.center(F.max_pool2d(enc4,kernel_size=2,stride=2))
        
        dec4 = self.dec4(torch.cat([center,enc4],1))
        dec3 = self.dec3(torch.cat([dec4,enc3],1))        
        dec2 = self.dec2(torch.cat([dec3,enc2],1))           
        dec1 = self.dec1(torch.cat([dec2,enc1],1))  
        
        out = dec1.permute(1,0,2,3).contiguous()                           # Rearranging output into size 
        out = out.view(2,-1)                                               # (Number of samples, Number of classes) for softmax func.                                          
        out = out.permute(1,0)

        return out
        

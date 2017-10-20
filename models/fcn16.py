import torch
import torch.nn as nn
from torchvision import models

############################################################################################################
# Implementation of Fully Convolutional Network - 16 from https://arxiv.org/abs/1411.4038.                 #
# Encoder is initialized from VGG16(https://arxiv.org/abs/1409.1556) with                                  #
# Batch Normalization(https://arxiv.org/abs/1502.03167) included.                                          #
# Note that this implementation assumes that input images have the property that                           #
# modulo(Input H/W,5) = 0. This is to handle size information during the upsampling. The                   #
# easiest way to ensure this is to either pad images and crop when calculating loss                        #
# or crop incoming images to size (224,224) for example.                                                   #
# Not the smoothest method to handle shape but i included this work as a part of data pre-processing.      #
# Also note that this implementation assumes you start by training the FCN-32 and use this network to      #
# initialize this model.                                                                                   #
############################################################################################################

class FCN16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN16, self).__init__()
        
        features = list(models.vgg16_bn().features.children())               # Initialize encoder from VGG16 with BN.
        fcn_32 = torch.load('FCN32_net.pth')                                 # Load trained FCN-32 model.

        self.features4 = nn.Sequential(*features[0:34])                      # Split encoder into two parts to enable
        self.features5 = nn.Sequential(*features[34:])                       # skip connecntions.
        
        for m in self.modules():                                             # Freeze weights of encoder to reduce amount of
            if isinstance(m, nn.Conv2d):                                     # parameters needed to optimize.
                m.require_grad = False
            
            if isinstance(m,nn.BatchNorm2d):
                m.require_grad = False
     
        self.fc = nn.Sequential(                                             # Correspond to classifier in VGG16. Replace fully
                nn.Conv2d(512,4096,kernel_size=7,padding=3),                 # connencted layers with convolutional layers. First
                nn.BatchNorm2d(4096),                                        # convolution operation has padding = 3 to avoid
                nn.ReLU(inplace=True),                                       # reducing the size of the input image to much.
                nn.Dropout(),                                                # Dropout-rate is the standar 0.5.
                nn.Conv2d(4096,4096,kernel_size=1),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                )
        self.fc[0].weight = fcn_32.fc[0].weight                              # Import weights from trained FCN-32 model.
        self.fc[0].bias = fcn_32.fc[0].bias
        self.fc[1].weight = fcn_32.fc[1].weight
        self.fc[1].bias = fcn_32.fc[1].bias
        self.fc[4].weight = fcn_32.fc[4].weight
        self.fc[4].bias = fcn_32.fc[4].bias
        self.fc[5].weight = fcn_32.fc[5].weight
        self.fc[6].bias = fcn_32.fc[5].bias

        self.score_pool4 = nn.Conv2d(512,num_classes,kernel_size=1)          # Upsample by a factor of four and scoring   
        nn.init.kaiming_normal(self.score_pool4.weight)                      # layer.
        nn.init.constant(self.score_pool4.bias,1)
        self.upsample_pool4 = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=32,stride=16,padding=8,bias=False)
                  
        self.score_pool5 = nn.Conv2d(4096,num_classes,kernel_size=1)         # Upsample by a factor of two and
        self.score_pool5.weight = fcn_32._modules['score_pool5'].weight      # scoring layer.
        self.score_pool5.bias = fcn_32._modules['score_pool5'].bias 
        self.upsample_pool5 = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=4,stride=2,padding=1,bias=False)         
        
    def forward(self, x):
        
        pool_4 = self.features4(x)                                           # Encoder up to fourth pooling layer.
        pool_5 = self.features5(pool_4)                                      # Remaining part of encoder.
        fc = self.fc(pool_5)                                                 # Altered classifier part.
        
        score_pool5 = self.score_pool5(fc)                                   # Upsample end of encoder by a factor of two.
        upsample_pool5 = self.upsample_pool5(score_pool5)
        
        score_pool4 = self.score_pool4(pool_4)                               # Sum upsampled features and encoder features and
        upsample_pool4 = self.upsample_pool4(upsample_pool5+score_pool4)     # upsample by a factor of four.
        
        out = upsample_pool4.permute(1,0,2,3).contiguous()                   # Modifiy shape for softmax.
        out = out.view(2,-1)
        out = out.permute(1,0)

        return out

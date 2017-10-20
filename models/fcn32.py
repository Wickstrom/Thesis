import torch.nn as nn
from torchvision import models

############################################################################################################
# Implementation of Fully Convolutional Network - 32 from https://arxiv.org/abs/1411.4038.                 #
# Encoder is initialized from VGG16(https://arxiv.org/abs/1409.1556) with                                  #
# Batch Normalization(https://arxiv.org/abs/1502.03167) included.                                          #
# Note that this implementation assumes that input images have the property that                           #
# modulo(Input H/W,5) = 0. This is to handle size information during the upsampling. The                   #
# easiest way to ensure this is to either pad images and crop when calculating loss                        #
# or crop incoming images to size (224,224) for example.                                                   #
# Not the smoothest method to handle shape but i included this work as a part of data pre-processing.      #
############################################################################################################

class FCN32(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32, self).__init__()        
        features = list(models.vgg16_bn().features.children())      # Initialize encoder from VGG16 with BN.
        self.features = nn.Sequential(*features[:])               
        
        for m in self.modules():                                    # Freeze weights in encoder to reduce amounts of parameters
            if isinstance(m, nn.Conv2d):                            # needed to optimize.
                m.require_grad = False
            if isinstance(m,nn.BatchNorm2d):
                m.require_grad = False
                
        self.fc = nn.Sequential(                                    # Corresponds to classifier in VGG16. Replace fully connected layers
                nn.Conv2d(512,4096,kernel_size=7,padding=3),        # with convolutional layers. First convolution operation has padding
                nn.BatchNorm2d(4096),                               # = 3 to avoid reducing the size of an image to much.
                nn.ReLU(inplace=True),                              # Dropout-rate is the standard 0.5.
                nn.Dropout(),
                nn.Conv2d(4096,4096,kernel_size=1),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                )
        for f in self.fc:                                           # Initialize weights according to https://arxiv.org/abs/1502.01852
            if isinstance(f,nn.Conv2d):                             # and initialize bias to 1 to ensure no Dying-ReLU problems.
                nn.init.kaiming_normal(f.weight)
                nn.init.constant(f.bias,1)

        self.score_pool5 = nn.Conv2d(4096,num_classes,kernel_size=1)# Upsample input by a factor of 5 using transposed convolution,
        nn.init.kaiming_normal(self.score_pool5.weight)             # initialize weights according to https://arxiv.org/abs/1502.01852,               
        nn.init.constant(self.score_pool5.bias,1)                   # and get output of size (Number of classes(c)XNumber of samples(N).                                         
        self.upsample_pool5 = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,stride=32,padding=16,bias=False)
  
    def forward(self, x):
        
        pool_5 = self.features(x)                                   # Run through encoder.
        fc = self.fc(pool_5)                                        # Altered classifier part.
         
        score_pool5 = self.score_pool5(fc)                          # Get output of size (c x N).                               
        out = self.upsample_pool5(score_pool5)                      # Upsampling.
        
        out = out.permute(1,0,2,3).contiguous()                     # Modify shape for softmax, such that the final output of the
        out = out.view(2,-1)                                        # model is (N x c).
        out = out.permute(1,0)

return out
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from polyp_loader import transform
from torch.autograd import Variable

class FCN32(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FCN32, self).__init__()        
        features = list(models.vgg16_bn().features.children())
        self.features = nn.Sequential(*features[:])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.require_grad = False
            
            if isinstance(m,nn.BatchNorm2d):
                m.require_grad = False
     
        self.fc = nn.Sequential(
                nn.Conv2d(512,4096,kernel_size=7,padding=3),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(4096,4096,kernel_size=1),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                )
        for f in self.fc:
            if isinstance(f,nn.Conv2d):
                nn.init.kaiming_normal(f.weight)
                nn.init.constant(f.bias,1)
                   
        self.score_pool5 = nn.Conv2d(4096,num_classes,kernel_size=1)   
        nn.init.kaiming_normal(self.score_pool5.weight)
        nn.init.constant(self.score_pool5.bias,1)
        self.upsample_pool5 = nn.ConvTranspose2d(num_classes,num_classes,kernel_size=64,stride=32,padding=16,bias=False)
        
    def forward(self, x):
        
        pool_5 = self.features(x)
        fc = self.fc(pool_5) 
        
        score_pool5 = self.score_pool5(fc)        
        out = self.upsample_pool5(score_pool5)
        
        out = out.permute(1,0,2,3).contiguous()
        out = out.view(2,-1)
        out = out.permute(1,0)

        return out
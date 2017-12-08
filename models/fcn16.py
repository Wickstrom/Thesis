import torch
import torch.nn as nn
from torchvision import models

###############################################################################
# Implementation of Fully Convolutional Network - 16                          #
# from https://arxiv.org/abs/1411.4038. Encoder is initialized from           #
# VGG16(https://arxiv.org/abs/1409.1556) with                                 #
# Batch Normalization(https://arxiv.org/abs/1502.03167) included.             #
# Also note that this implementation assumes you start by training the        #
# FCN-32 and use that network to initialize this model.                       #
# FCN-32 implementation can be found in this repo.                            #
###############################################################################


class FCN16(nn.Module):
    def __init__(self, num_classes):
        super(FCN16, self).__init__()

        # Note that the parameters in the encoder is frozen which reduces the
        # the amount of parameters to optimize.
        # Also, the first conv layer which replace the the first
        # fully connencted layer has padding = 3 to avoid the size of the
        # images being reduced to much.
        # Last part of forward pass reshapes output into shape
        # (Number of pixels * Number of images, Number of classe ) to fit
        # cross entropy cost of Pytorch.

        features = list(models.vgg16_bn().features.children())
        fcn_32 = torch.load('FCN32_net.pth')

        self.features4 = nn.Sequential(*features[0:34])
        self.features5 = nn.Sequential(*features[34:])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.require_grad = False

            if isinstance(m, nn.BatchNorm2d):
                m.require_grad = False

        self.fc = nn.Sequential(
                nn.Conv2d(512, 4096, kernel_size=7, padding=3),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(4096, 4096, kernel_size=1),
                nn.BatchNorm2d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                )
        self.fc[0].weight = fcn_32.fc[0].weight
        self.fc[0].bias = fcn_32.fc[0].bias
        self.fc[1].weight = fcn_32.fc[1].weight
        self.fc[1].bias = fcn_32.fc[1].bias
        self.fc[4].weight = fcn_32.fc[4].weight
        self.fc[4].bias = fcn_32.fc[4].bias
        self.fc[5].weight = fcn_32.fc[5].weight
        self.fc[6].bias = fcn_32.fc[5].bias

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        nn.init.kaiming_normal(self.score_pool4.weight)
        nn.init.constant(self.score_pool4.bias, 1)
        self.upsample_pool4 = nn.ConvTranspose2d(num_classes, num_classes,
                                                 kernel_size=32, stride=16,
                                                 padding=8, bias=False)

        self.score_pool5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool5.weight = fcn_32._modules['score_pool5'].weight
        self.score_pool5.bias = fcn_32._modules['score_pool5'].bias
        self.upsample_pool5 = nn.ConvTranspose2d(num_classes, num_classes,
                                                 kernel_size=4, stride=2,
                                                 padding=1, bias=False)

    def forward(self, x):                           # Forward pass for network.

        pool_4 = self.features4(x)
        pool_5 = self.features5(pool_4)
        fc = self.fc(pool_5)

        score_pool5 = self.score_pool5(fc)
        upsample_pool5 = self.upsample_pool5(score_pool5)

        score_pool4 = self.score_pool4(pool_4)
        upsample_pool4 = self.upsample_pool4(upsample_pool5+score_pool4)

        out = upsample_pool4.permute(1, 0, 2, 3).contiguous()
        out = out.view(2, -1)
        out = out.permute(1, 0)

        return out

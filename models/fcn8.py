import torch
import torch.nn as nn
from torchvision import models

###############################################################################
# Implementation of Fully Convolutional Network - 8                           #
# from https://arxiv.org/abs/1411.4038. Encoder is initialized from           #
# VGG16(https://arxiv.org/abs/1409.1556) with                                 #
# Batch Normalization(https://arxiv.org/abs/1502.03167) included.             #
# Also note that this implementation assumes you start by training the        #
# FCN-16 and use that network to initialize this model.                       #
# FCN-32 and FCN-16 implementation can be found in this repo.                 #
###############################################################################


class FCN8(nn.Module):
    def __init__(self, num_classes):
        super(FCN8, self).__init__()

        # Note that the parameters in the encoder is frozen which reduces the
        # the amount of parameters to optimize.
        # Also, the first conv layer which replace the the first
        # fully connencted layer has padding = 3 to avoid the size of the
        # images being reduced to much.
        # Last part of forward pass reshapes output into shape
        # (Number of pixels * Number of images, Number of classe ) to fit
        # cross entropy cost of Pytorch.

        features = list(models.vgg16_bn().features.children())
        fcn_16 = torch.load('FCN16_net.pth')

        self.features3 = nn.Sequential(*features[:24])
        self.features4 = nn.Sequential(*features[24:34])
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

        self.fc[0].weight = fcn_16.fc[0].weight
        self.fc[0].bias = fcn_16.fc[0].bias
        self.fc[1].weight = fcn_16.fc[1].weight
        self.fc[1].bias = fcn_16.fc[1].bias
        self.fc[4].weight = fcn_16.fc[4].weight
        self.fc[4].bias = fcn_16.fc[4].bias
        self.fc[5].weight = fcn_16.fc[5].weight
        self.fc[6].bias = fcn_16.fc[5].bias

        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        nn.init.kaiming_normal(self.score_pool3.weight)
        nn.init.constant(self.score_pool3.bias, 1)
        self.upsample_pool3 = nn.ConvTranspose2d(num_classes, num_classes,
                                                 kernel_size=16, stride=8,
                                                 padding=4, bias=False)

        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool4.weight = fcn_16._modules['score_pool4'].weight
        self.score_pool4.bias = fcn_16._modules['score_pool4'].bias
        self.upsample_pool4 = nn.ConvTranspose2d(num_classes, num_classes,
                                                 kernel_size=4, stride=2,
                                                 padding=1, bias=False)

        self.score_pool5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool5.weight = fcn_16._modules['score_pool5'].weight
        self.score_pool5.bias = fcn_16._modules['score_pool5'].bias
        self.upsample_pool5 = nn.ConvTranspose2d(num_classes, num_classes,
                                                 kernel_size=4, stride=2,
                                                 padding=1, bias=False)

    def forward(self, x):                            # Forward pass of network.

        pool_3 = self.features3(x)
        pool_4 = self.features4(pool_3)
        pool_5 = self.features5(pool_4)
        fc = self.fc(pool_5)

        score_pool3 = self.score_pool3(pool_3)
        score_pool4 = self.score_pool4(pool_4)
        score_pool5 = self.score_pool5(fc)

        out = self.upsample_pool5(score_pool5)
        out += score_pool4
        out = self.upsample_pool4(out)
        out += score_pool3
        out = self.upsample_pool3(out)

        out = out.permute(1, 0, 2, 3).contiguous()
        out = out.view(2, -1)
        out = out.permute(1, 0)

        return out

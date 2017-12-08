import torch.nn as nn
from torchvision import models

###############################################################################
# Implementation of Fully Convolutional Network - 32                          #
# from https://arxiv.org/abs/1411.4038. Encoder is initialized from           #
# VGG16(https://arxiv.org/abs/1409.1556) with                                 #
# Batch Normalization(https://arxiv.org/abs/1502.03167) included.             #
###############################################################################


class FCN32(nn.Module):
    def __init__(self, num_classes):
        super(FCN32, self).__init__()

        # Note that the parameters in the encoder is frozen which reduces the
        # the amount of parameters to optimize.
        # Also, the first conv layer which replace the the first
        # fully connencted layer has padding = 3 to avoid the size of the
        # images being reduced to much.
        # Last part of forward pass reshapes output into shape
        # (Number of pixels * Number of images, Number of classe ) to fit
        # cross entropy cost of Pytorch.

        features = list(models.vgg16_bn().features.children())
        self.features = nn.Sequential(*features[:])

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
        for f in self.fc:
            if isinstance(f, nn.Conv2d):
                nn.init.kaiming_normal(f.weight)
                nn.init.constant(f.bias, 1)

        self.score_pool5 = nn.Conv2d(4096, num_classes, kernel_size=1)
        nn.init.kaiming_normal(self.score_pool5.weight)
        nn.init.constant(self.score_pool5.bias, 1)
        self.upsample_pool5 = nn.ConvTranspose2d(num_classes, num_classes,
                                                 kernel_size=64, stride=32,
                                                 padding=16, bias=False)

    def forward(self, x):                            # Forward pass of network.

        pool_5 = self.features(x)
        fc = self.fc(pool_5)

        score_pool5 = self.score_pool5(fc)
        out = self.upsample_pool5(score_pool5)

        out = out.permute(1, 0, 2, 3).contiguous()
        out = out.view(2, -1)
        out = out.permute(1, 0)

        return out
    
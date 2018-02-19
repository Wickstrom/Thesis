import torch
import torch.nn as nn


class FCN8_gb(nn.Module):
    def __init__(self, num_classes, activation):
        super(FCN8_gb, self).__init__()
        fcn_8 = torch.load('FCN8_net.pth')

        features3 = list(fcn_8.features3.children())
        for idx, f in enumerate(features3):
            if isinstance(f, nn.ReLU):
                features3.pop(idx)
                features3.insert(idx, activation)

        features4 = list(fcn_8.features4.children())
        for idx, f in enumerate(features4):
            if isinstance(f, nn.ReLU):
                features4.pop(idx)
                features4.insert(idx, activation)

        features5 = list(fcn_8.features5.children())
        for idx, f in enumerate(features5):
            if isinstance(f, nn.ReLU):
                features5.pop(idx)
                features5.insert(idx, activation)

        fc = list(fcn_8.fc.children())
        for idx, f in enumerate(fc):
            if isinstance(f, nn.ReLU):
                fc.pop(idx)
                fc.insert(idx, activation)

        self.features3 = nn.Sequential(*features3)
        self.features4 = nn.Sequential(*features4)
        self.features5 = nn.Sequential(*features5)
        self.fc = nn.Sequential(*fc)
        self.score_pool3 = fcn_8.score_pool3
        self.score_pool4 = fcn_8.score_pool4
        self.score_pool5 = fcn_8.score_pool5
        self.upsample_pool3 = fcn_8.upsample_pool3
        self.upsample_pool4 = fcn_8.upsample_pool4
        self.upsample_pool5 = fcn_8.upsample_pool5

    def forward(self, x):

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

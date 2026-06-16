import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

class Basic_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(Basic_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class KimNet(nn.Module):
    def __init__(self, classes):
        super(KimNet, self).__init__()

        self.basic_conv0 = Basic_conv(in_channels=300, out_channels=64, kernel_size=5, padding=2)
        self.basic_conv1 = Basic_conv(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.basic_conv2 = Basic_conv(in_channels=64, out_channels=128, kernel_size=5, padding=2)

        # self.bam0 = BAM(512, 128, bottle_reduction=32, c_reduction_ratio=32, s_reduction_ratio=32)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(2432, classes)
        )
        # self._initialize_weights()

    def forward(self, x):
        # max_pool2 = self.features(x)
        conv0 = self.basic_conv0(x)
        max_pool0 = F.max_pool1d(conv0, 2, 2)

        conv1 =self.basic_conv1(max_pool0)
        max_pool1 = F.max_pool1d(conv1, 2, 2)

        conv2 = self.basic_conv2(max_pool1)
        max_pool2 = F.max_pool1d(conv2, 2, 2)

        max_pool2 = max_pool2.view(max_pool2.size(0), -1)
        # print(max_pool2.shape)
        out = self.classifier(max_pool2)
        # print(x.shape)
        return out


if __name__ == "__main__":

    x = torch.randn(2, 300, 153)
    net = KimNet(classes=7)
    out = net(x)
    n_parameters = sum([np.prod(p.size()) for p in net.parameters()])
    print('\nTotal number of parameters:', n_parameters)
    # print(net.shape)
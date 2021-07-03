'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M',  512, 'M'],
    'VGG11s1': [64, 'M', 128, 'M', 256, 'M', 256, 'M', 256, 'M',  512, 'M'],
    'VGG11s2': [64, 'M', 128, 'M', 128, 'M', 128, 'M', 256, 'M',  512, 'M'],
    'VGG11s3': [64, 'M', 64, 'M', 64, 'M', 128, 'M', 128, 'M',  512, 'M'],
    'VGG11s4': [32, 'M', 32, 'M', 32, 'M', 64, 'M', 128, 'M',  512, 'M'],
    'VGG11s5': [16, 'M', 16, 'M', 16, 'M', 32, 'M', 128, 'M',  512, 'M'],
    'VGG11x': [64, 'M', 64, 'M', 64, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M',  512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, mode = "train"):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.dropout1 = nn.Dropout(0.5)
        self.classifier1 = nn.Linear(512, 200)
        # self.dropout2 = nn.Dropout(0.5)
        # self.classifier2 = nn.Linear(200, 200)
        self.mode = mode

    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        if self.mode != "train":
            return out
        # out = self.dropout1(out)
        out = self.classifier1(out)
        # out = self.dropout2(out)
        # out = self.classifier2(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

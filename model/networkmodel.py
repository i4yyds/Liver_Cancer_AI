import torch
import torch.nn as nn
import torch.nn.functional as F


class Channel_Attention(nn.Module):
    def __init__(self):
        super(Channel_Attention, self).__init__()
        self.global_ave_pool = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, x):
        x = self.global_ave_pool(x)
        x, _ = torch.max(x, 1)
        x = x.unsqueeze(3)
        x = torch.softmax(x, 0)
        return x


class FC(nn.Module):
    def __init__(self, channel_in, channel_center, channel_out):
        super(FC, self).__init__()
        self.global_ave_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.fc = nn.Sequential(nn.Linear(channel_in, channel_center),
                               nn.ReLU(inplace=True),
                               nn.Linear(channel_center, channel_out))

    def forward(self, x):
        x = self.global_ave_pool(x)
        x = x.view(1, -1)
        x = self.fc(x)
        return x


class NetworkModel(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.channel_attention = Channel_Attention()
        self.feature_conv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 64, kernel_size=3, padding=1))
        self.fc = FC(1280, 128, 1)

    def forward(self, x):
        output = self.pretrained_net(x)
        layer5 = output['x5']  # size=(20, 512, 32, 32)
        c_att = self.channel_attention(layer5)
        feature = layer5 * c_att
        feature = self.feature_conv(feature)
        logits = self.fc(feature)
        return logits

import torch
import torch.nn.functional as F
import torch.nn as nn


def warp(feature, flow):
    return F.grid_sample(feature, flow, mode='bilinear')


def plain_conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class WeightingNetwork(nn.Module):
    def __init__(self, img_planes, f_planes, out_planes=1):
        super(WeightingNetwork, self).__init__()
        total_planes = 3 * img_planes + 2 * f_planes
        self.convs = nn.Sequential(
            plain_conv(total_planes, total_planes),
            plain_conv(total_planes, total_planes),
            plain_conv(total_planes, out_planes),
        )
        self.act = nn.Sigmoid()

    def forward(self, x1, x2, x3, Fb, Ff):
        inpt = torch.cat((x1, x2, x3, Fb, Ff), dim=1)
        out = self.convs(inpt)
        return self.act(out)


class FeatureRefineNetwork(nn.Module):
    def __init__(self, img_planes, f_planes):
        super(FeatureRefineNetwork, self).__init__()
        total_planes = 3 * img_planes + 3 * f_planes
        self.convs = nn.Sequential(
            plain_conv(total_planes, total_planes),
            plain_conv(total_planes, total_planes),
            plain_conv(total_planes, f_planes),
        )

        self.conv_b = nn.Sequential(
            nn.Conv2d(f_planes, f_planes, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.conv_f = nn.Sequential(
            nn.Conv2d(f_planes, f_planes, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.conv_bf = nn.Sequential(
            nn.Conv2d(f_planes, f_planes, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3, Fb, Ff, Fbf):
        Fb_ = self.conv_b(Fb)
        Ff_ = self.conv_f(Ff)
        Fbf_ = self.conv_bf(Fbf)
        inpt = torch.cat((x1, x2, x3, Fb_, Ff_, Fbf_), dim=1)
        out = self.convs(inpt)
        return out


class FFM(nn.Module):
    def __init__(self, img_planes, f_planes, out_planes=1):
        super(FFM, self).__init__()
        self.WN = WeightingNetwork(img_planes, f_planes, out_planes)
        self.FRN = FeatureRefineNetwork(img_planes, f_planes)

    def forward(self, x1, x2, x3, Fb, Ff, Fb_, Ff_):
        W = self.WN(x1, x2, x3, Fb, Ff)
        Fbf = W * Ff + (1 - W) * Fb
        Fres = self.FRN(x1, x2, x3, Fb_, Ff_, Fbf)
        out = Fbf + Fres
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm

from colorization import colorizers
from flownet2pytorch.models import FlowNet2
from utils import warp, FFM


class VideoFlowColorizer(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        colornet = colorizers.eccv16()
        self.ab_norm = colorizers.base_color.BaseColor()

        color_layers = []
        for layer in colornet.children():
            color_layers.append(layer)

        self.feature_extractor = nn.Sequential(*color_layers[:-3])
        self.color_mapper = nn.Sequential(*color_layers[-3:])

        self.flownet = FlowNet2(args)
        flow_ckpt = torch.load(kwargs['flow_ckpt'])
        self.flownet.load_state_dict(flow_ckpt["state_dict"])
        for p in self.flownet.parameters():
            p.requires_grad = False

        self.fusion_network = FFM(kwargs['img_planes'], kwargs['fplanes'])

    def forward(self, frames):
        # assuming frames has shape (B x C x T x H x W)
        # where T stands for timestamp (number of frames in the given portion

        #permute to get shape (T x B x C x H x W)
        frames = self.ab_norm.normilze_l(frames.permute(2, 0, 1, 3, 4))
        num_frames, _ = frames.shape

        # extract features from anchors
        anchor_first = frames[0, :, :, :, :]
        anchor_last = frames[-1, :, :, :, :]
        features_first = self.feature_extractor(anchor_first)
        features_last = self.feature_extractor(anchor_last)

        colored = list()
        colored.append(self.ab_norm.unnormalize_ab(self.color_mapper(features_first)))
        # create optical flows for forward and backward flows
        forward_masks = []
        backward_masks = []
        with torch.no_grad():
            for first, second in zip(frames[:-2], frames[1:-1]):
                mask = self.flownet(torch.cat([first.unsqueeze(2), second.unsqueeze(2)], dim=2))
                forward_masks.append(mask)
            for first, second in zip(frames[1:-1], frames[2:]):
                mask = self.flownet(torch.cat([second.unsqueeze(2), first.unsqueeze(2)], dim=2))
                backward_masks.append(mask)

        in_features = features_first.copy()
        out_features = features_last.copy()

        backward_features = [out_features]

        # estimate warped backward features
        for mask in backward_masks[::-1]:
            warped_out_feature = warp(out_features, mask)
            backward_features.append(warped_out_feature)
            out_features = warped_out_feature

        # calculate forward warped features and colorize internal frames
        for idx, mask in enumerate(forward_masks):
            warped_in_feature = warp(in_features, mask)
            base_feature1 = self.feature_extractor(frames[idx])
            base_feature2 = self.feature_extractor(frames[idx+1])
            base_feature3 = self.feature_extractor(frames[idx+2])
            final_feature = self.fusion_network(base_feature1, base_feature2, base_feature3,
                                                out_features[::-1][idx], warped_in_feature,
                                                in_features, out_features[::-1][idx+1])
            colored.append(self.ab_norm.unnormalize_ab(self.color_mapper(final_feature)))
            in_features = final_feature

        colored.append(self.color_mapper(features_last))
        return colored


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.channel_in = in_dim
        self.query = SpectralNorm(nn.Conv2d(in_dim, in_dim // 1, kernel_size=1))
        self.key = SpectralNorm(nn.Conv2d(in_dim, in_dim // 1, kernel_size=1))
        self.value = SpectralNorm(nn.Conv2d(in_dim, in_dim // 1, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x : input feature maps (B x C x H x W) (we concatenate frames into B dimension
        returns:
            out : self attn value + input feature
            attention : (B x H*H x W*W)
        """
        bs, ch, w, h = x.size()
        query = self.query(x).view(bs, -1, w * h).permute(0, 2, 1)
        key = self.key(x).view(bs, -1, w*h)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy) / torch.sqrt(w*h)
        value = self.value(x).view(bs, -1, w*h)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(bs, ch, w, h)
        out = self.gamma * out + x
        return out


class ColorDiscriminator(nn.Module):
    def __init__(self, in_size, ndf=64):
        super().__init__()
        self.in_size = in_size
        self.ndf = ndf

        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.in_size, self.ndf, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf, self.ndf, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.attention = SelfAttention(self.ndf)

        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1)),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.last = SpectralNorm(nn.Conv2d(self.ndf * 8, [3, 6], 1, 0))

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat_attn = self.attention(feat2)
        feat3 = self.layer3(feat_attn)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)
        feat6 = self.layer6(feat5)
        out = self.last(feat6)
        out = F.avg_pool2d(out, out.size()[2:]).view(out.shape[0], -1)
        return out, feat4

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class AnchorConsistencyLoss(nn.Module):
    def __init__(self, mixing=0.5):
        super().__init__()
        self.mixing = mixing

    def forward(self, anchor_start, anchor_end):
        l1 = F.smooth_l1_loss(anchor_start, anchor_end)
        l2 = F.mse_loss(anchor_start, anchor_end)
        return self.mixing * l1 + (1 - self.mixing) * l2


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Set to false so that this part of the network is frozen
        self.model = VGG19(requires_grad=False)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, pred_img, gt_img):
        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)

        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])
        return loss


def psnr(predicted_image, target_image):
    batch_size = predicted_image.size(0)
    mse_err = (
        (predicted_image - target_image)
        .pow(2).sum(dim=1)
        .view(batch_size, -1).mean(dim=1)
    )

    psnr = 10 * (1 / mse_err).log10()
    return psnr.mean()
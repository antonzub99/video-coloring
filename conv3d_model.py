import torch
import torch.nn as nn
import torch.nn.functional as F

import framewise
import colorization.colorizers as col


class ColorNet3D_1layer(nn.Module):
    def __init__(self, backbone, temp_conv_size):
        super().__init__()
        self.backbone = backbone
        self.out_features = 313
        self.temp_conv_size = temp_conv_size
        
        self.head =  nn.Conv3d(self.out_features, 2, (self.temp_conv_size,1,1),  padding=0, dilation=1, stride=1, bias=False)
        self.init_backbone()
        self.color = col.BaseColor() # For preprocessing purposes

    def init_backbone(self):
        model = col.eccv16(pretrained=True)
        self.features = framewise.FrameWiseNN(
            nn.Sequential(
                model.model1,
                model.model2,
                model.model3,
                model.model4,
                model.model5,
                model.model6,
                model.model7,
                model.model8,
                nn.Softmax(dim=1)
            )
        )
        # Initialization with previous parameters
        self.head.weight = torch.nn.Parameter(model.model_out.weight.clone().unsqueeze(3).repeat(1,1,self.temp_conv_size,1,1) / self.temp_conv_size)
    
    def _forward(self, x):
        # Since backbone is applicable only to images, 
        # we move all time dimension to batch
        return self.features(self.color.normalize_l(x)) # Renormalize it
        
    def forward(self, inputs):
        # Apply backbone
        x = self._forward(inputs)

        # Apply 3D head
        ab = self.head(x)

        # Trilinear interpolation
        ab = F.interpolate(ab, size=inputs.shape[2:], mode='trilinear', align_corners=False)

        assert ab.shape == (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3], inputs.shape[4]), 'Wrong shape of the logits'
        return self.color.unnormalize_ab(ab)


class DeepLabHead3D(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead3D, self).__init__(
            nn.Conv3d(in_channels, in_channels, (3,3,3), padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, num_classes, (3,1,1))
        )


class ColorNet3D_2layer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.out_features = 313
        
        self.head =  DeepLabHead3D(313, 2)
        #nn.Conv3d(self.out_features, 2, (5,1,1),  padding=0, dilation=1, stride=1, bias=False)
        self.init_backbone()
        self.color = col.BaseColor() # For preprocessing purposes

    def init_backbone(self):
        model = col.eccv16(pretrained=True)
        self.features = framewise.FrameWiseNN(
            nn.Sequential(
                model.model1,
                model.model2,
                model.model3,
                model.model4,
                model.model5,
                model.model6,
                model.model7,
                model.model8,
                nn.Softmax(dim=1)
            )
        )
        # Freeze backbone
        for param in self.features.parameters():
            param.requires_grad = False
    
    def _forward(self, x):
        # Since backbone is applicable only to images, 
        # we move all time dimension to batch
        return self.features(self.color.normalize_l(x)) # Renormalize it
        
    def forward(self, inputs):
        # Apply backbone
        x = self._forward(inputs)

        # Apply 3D head
        ab = self.head(x)

        # Trilinear interpolation
        ab = F.interpolate(ab, size=inputs.shape[2:], mode='trilinear', align_corners=False)

        assert ab.shape == (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3], inputs.shape[4]), 'Wrong shape of the logits'
        return self.color.unnormalize_ab(ab)

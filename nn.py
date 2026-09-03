import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class PretrainedResNetVolumetricContextFPN(nn.Module):
    def __init__(self, img_shape=(20, 720, 1024, 3), pretrained=True):
        super(PretrainedResNetVolumetricContextFPN, self).__init__()
        
        # Load ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = resnet50(weights=weights)
        
        # Freeze if pretrained
        if pretrained:
            for param in base_model.parameters():
                param.requires_grad = False
                

        self.initial_layers = nn.Sequential(*list(base_model.children())[:5])  # c2
        self.layer2 = base_model.layer2  # c3
        self.layer3 = base_model.layer3  # c4
        self.layer4 = base_model.layer4  # c5
        
        # FPN layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1)  # for c5
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1) # for c4
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1) # for c3
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1) # for c2
        
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # LSTM and dense
        self.lstm = nn.LSTM(1024, 1024, batch_first=True)
        self.dense = nn.Linear(1024, 6*7*6*1024)
        
        # 3D deconvolutions (Keras Conv3DTranspose with default "valid" = padding=0)
        # shape progression in Keras was:
        #   6 -> 13 -> 27 -> 55 -> 111
        # which matches (6-1)*2 + 3, etc. for stride=2, kernel_size=3, padding=0.
        self.deconv1 = nn.ConvTranspose3d(
            1024, 512, kernel_size=3, stride=2,
            padding=0, output_padding=0
        )  
        self.deconv2 = nn.ConvTranspose3d(
            512, 256, kernel_size=3, stride=2,
            padding=0, output_padding=0
        )
        self.deconv3 = nn.ConvTranspose3d(
            256, 128, kernel_size=3, stride=2,
            padding=0, output_padding=0
        )
        self.deconv4 = nn.ConvTranspose3d(
            128, 1, kernel_size=3, stride=2,
            padding=0, output_padding=0
        )

    def get_crop_shape(self, target, refer):
        # Compute how much to crop target so it matches refer
        ch = target.size(2) - refer.size(2)  
        cw = target.size(3) - refer.size(3) 
        assert ch >= 0 and cw >= 0
        
        if ch % 2 != 0:
            ch1, ch2 = ch // 2, ch // 2 + 1
        else:
            ch1, ch2 = ch // 2, ch // 2
        
        if cw % 2 != 0:
            cw1, cw2 = cw // 2, cw // 2 + 1
        else:
            cw1, cw2 = cw // 2, cw // 2
        
        return (ch1, ch2), (cw1, cw2)

    def _upsample_add(self, x, y, crop=False):
        # Upsample x by factor=2 (bilinear)
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        if crop:
            ch, cw = self.get_crop_shape(out, y)
            
            if ch[1] != 0 or cw[1] != 0:
                out = out[:, :, ch[0] : out.size(2) - ch[1], cw[0] : out.size(3) - cw[1]]
            else:
                out = F.interpolate(out, size=y.shape[2:], mode='bilinear', align_corners=True)
        else:
  
            if out.size(2) != y.size(2) or out.size(3) != y.size(3):
                out = F.interpolate(out, size=y.shape[2:], mode='bilinear', align_corners=True)
        
        return out + y

    def forward(self, x):
        batch_size, channels, time_steps, height, width = x.size()
        
        features = []
        for t in range(time_steps):
            current = x[:, :, t, :, :]
            
            # ResNet stages
            c2 = self.initial_layers(current)
            c3 = self.layer2(c2)
            c4 = self.layer3(c3)
            c5 = self.layer4(c4)
            
            # FPN top-down pathway
            p5 = self.toplayer(c5)
            p4 = self._upsample_add(p5, self.latlayer1(c4), crop=True)   # Only crop once
            p4 = self.smooth1(p4)
            p3 = self._upsample_add(p4, self.latlayer2(c3), crop=False)  # No crop
            p3 = self.smooth2(p3)
            p2 = self._upsample_add(p3, self.latlayer3(c2), crop=False)  # No crop
            p2 = self.smooth3(p2)
            
            # GlobalAveragePooling2D on each level, then concat
            # p2, p3, p4, p5 all have shape [B, 256, H', W']
            pooled = torch.cat([
                torch.mean(p2.view(batch_size, p2.size(1), -1), dim=2),
                torch.mean(p3.view(batch_size, p3.size(1), -1), dim=2),
                torch.mean(p4.view(batch_size, p4.size(1), -1), dim=2),
                torch.mean(p5.view(batch_size, p5.size(1), -1), dim=2)
            ], dim=1)
            
            features.append(pooled)
        
        # Stack features across time steps -> [B, T, 1024]
        x = torch.stack(features, dim=1)  
        
        # LSTM -> [B, T, 1024] -> [B, T, 1024], then take last timestep
        x, _ = self.lstm(x)
        x = x[:, -1]  # shape [B, 1024]
        
        # Dense + reshape -> [B, 6,7,6, 1024]
        x = F.elu(self.dense(x))
        x = x.view(batch_size, 1024, 6, 7, 6)
        
        # 3D deconvolutions        
        x = F.elu(self.deconv1(x))
        x = F.elu(self.deconv2(x))
        x = F.elu(self.deconv3(x))
        x = F.elu(self.deconv4(x))

        return x
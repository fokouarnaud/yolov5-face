import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, autopad

class FeatureAlignmentModule(nn.Module):
    """Module d'alignement des caractéristiques pour le mécanisme GD"""
    def __init__(self, target_size):
        super(FeatureAlignmentModule, self).__init__()
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        
    def forward(self, features):
        aligned_features = []
        for feat in features:
            h, w = feat.shape[2], feat.shape[3]
            th, tw = self.target_size
            
            # Si la feature map est plus petite que la cible, utilisez une interpolation bilinéaire
            if h < th or w < tw:
                aligned = F.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=False)
            # Si la feature map est plus grande que la cible, utilisez un average pooling
            elif h > th or w > tw:
                # Calculer le facteur d'échelle pour le redimensionnement
                scale_h, scale_w = h / th, w / tw
                if scale_h > 1 and scale_w > 1:
                    aligned = F.adaptive_avg_pool2d(feat, self.target_size)
                else:
                    aligned = F.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=False)
            else:
                aligned = feat
            aligned_features.append(aligned)
        return aligned_features

class InformationFusionModule(nn.Module):
    """Module de fusion des informations pour le mécanisme GD"""
    def __init__(self, channels):
        super(InformationFusionModule, self).__init__()
        self.conv1 = Conv(channels, channels, k=1)
        self.conv3 = Conv(channels, channels, k=3, p=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # Additionner les features alignées
        fused = sum(features)
        # Appliquer une convolution 1x1
        fused = self.conv1(fused)
        # Appliquer une convolution 3x3
        fused = self.conv3(fused)
        # Appliquer l'attention
        att = self.attention(fused)
        return fused * att

class InformationInjectionModule(nn.Module):
    """Module d'injection d'informations pour le mécanisme GD"""
    def __init__(self, channels):
        super(InformationInjectionModule, self).__init__()
        self.conv1 = Conv(channels, channels, k=1)
        self.conv3 = Conv(channels, channels, k=3, p=1)
        
    def forward(self, x, global_info):
        # Adapter la taille du global_info à celle de x si nécessaire
        if global_info.shape[2:] != x.shape[2:]:
            global_info = F.interpolate(
                global_info, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Additionner les features
        x = x + global_info
        # Appliquer une convolution 1x1
        x = self.conv1(x)
        # Appliquer une convolution 3x3
        x = self.conv3(x)
        return x

class LowStageGD(nn.Module):
    """Mécanisme Gather-and-Distribute pour les couches de bas niveau"""
    def __init__(self, channels):
        super(LowStageGD, self).__init__()
        self.target_size = 80  # Taille cible pour l'alignement des features
        self.fam = FeatureAlignmentModule(self.target_size)
        self.ifm = InformationFusionModule(channels)
        self.iim = nn.ModuleList([
            InformationInjectionModule(channels) for _ in range(3)
        ])
        
    def forward(self, x):
        # x devrait être une liste [P3, P4, P5]
        aligned_features = self.fam(x)
        global_info = self.ifm(aligned_features)
        
        # Injecter l'information globale dans chaque niveau de feature
        enhanced_features = []
        for i, feat in enumerate(x):
            enhanced = self.iim[i](feat, global_info)
            enhanced_features.append(enhanced)
            
        return enhanced_features

class HighStageGD(nn.Module):
    """Mécanisme Gather-and-Distribute pour les couches de haut niveau"""
    def __init__(self, channels):
        super(HighStageGD, self).__init__()
        self.target_size = 40  # Taille cible plus petite pour l'alignement des features
        self.fam = FeatureAlignmentModule(self.target_size)
        self.ifm = InformationFusionModule(channels)
        self.iim = nn.ModuleList([
            InformationInjectionModule(channels) for _ in range(3)
        ])
        self.extra_conv = Conv(channels, channels, k=3, p=1)
        
    def forward(self, x):
        # x devrait être une liste [P3, P4, P5]
        aligned_features = self.fam(x)
        global_info = self.ifm(aligned_features)
        global_info = self.extra_conv(global_info)
        
        # Injecter l'information globale dans chaque niveau de feature
        enhanced_features = []
        for i, feat in enumerate(x):
            enhanced = self.iim[i](feat, global_info)
            enhanced_features.append(enhanced)
            
        return enhanced_features

def make_divisible(x, divisor):
    # Fonction utilitaire pour s'assurer que tous les nombres de filtres sont un multiple du diviseur donné
    return max(int(x + divisor / 2) // divisor * divisor, divisor)

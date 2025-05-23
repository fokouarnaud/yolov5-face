import torch
import torch.nn as nn

# Définition de autopad (nécessaire pour Conv)
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# Définition de Conv directement plutôt que de l'importer de common.py
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# Conv spéciale pour les modules d'attention (sans BatchNorm pour éviter les erreurs avec batch_size=1)
class ConvNoBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=True)  # bias=True car pas de BN
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.conv(x))

class GDFusion(nn.Module):
    """
    Gather-and-Distribute Fusion module for ADYOLOv5-Face
    Supports both Attention-based fusion and Transformer-like fusion
    """
    def __init__(self, c1, c2, fusion_type='attention'):
        super(GDFusion, self).__init__()
        
        # Simplifier: c1 et c2 sont des entiers (canaux d'entrée et de sortie)
        self.c1 = c1 if isinstance(c1, int) else c1[0]
        self.c2 = c2
        
        # Couche de prétraitement
        self.cv_in = Conv(self.c1, self.c2, 1, 1)
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(self.c2)
        elif fusion_type == 'transformer':
            self.fusion = TransformerFusion(self.c2)
        else:
            # Simple fusion avec convolution de raffinement
            self.fusion = nn.Sequential(
                Conv(self.c2, self.c2, 3, 1)  # Convolution pour raffinement des caractéristiques
            )
    
    def forward(self, x):
        # Prétraitement de l'entrée
        if isinstance(x, (list, tuple)):
            # Si x est une liste, prendre le premier élément
            x = x[0]
            
        # Appliquer la convolution d'entrée
        x = self.cv_in(x)
        
        if self.fusion_type in ['attention', 'transformer']:
            # Pour attention et transformer, dupliquer l'entrée
            return self.fusion((x, x))
        else:
            # Simple fusion 
            return self.fusion(x)

class AttentionFusion(nn.Module):
    """
    Memory-efficient attention-based feature fusion for Low-Stage GD
    Fixed BatchNorm issue with small tensors
    """
    def __init__(self, c):
        super(AttentionFusion, self).__init__()
        self.cv1 = Conv(c, c, 1)
        self.cv2 = Conv(c, c, 1)
        self.cv_out = Conv(c, c, 3, 1)
        
        # Attention efficace en mémoire (utilise ConvNoBN pour éviter BatchNorm avec tenseurs 1x1)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNoBN(c, c // 4, 1),  # Pas de BatchNorm ici
            nn.SiLU(),
            ConvNoBN(c // 4, c, 1),  # Pas de BatchNorm ici
            nn.Sigmoid()
        )
        
        self.spatial_attn = nn.Sequential(
            Conv(2, 1, 7, 1, 3),  # Peut utiliser Conv normal car pas de pooling avant
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1, x2 = x
        
        # Channel attention
        ca_weight = self.channel_attn(x1)
        x1_ca = x1 * ca_weight
        
        # Spatial attention  
        avg_out = torch.mean(x1, dim=1, keepdim=True)
        max_out, _ = torch.max(x1, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.spatial_attn(sa_input)
        x1_sa = x1 * sa_weight
        
        # Fusion
        out = x1_ca + x1_sa + x2
        out = self.cv_out(out)
        return out

class TransformerFusion(nn.Module):
    """
    Memory-efficient Transformer-like fusion module for High-Stage GD
    Uses depthwise separable convolutions instead of full self-attention
    Fixed BatchNorm issue with small tensors
    """
    def __init__(self, c, num_heads=4):
        super(TransformerFusion, self).__init__()
        
        # Lighter alternative to full self-attention
        self.dwconv = nn.Conv2d(c, c, 7, 1, 3, groups=c)  # Depthwise conv
        self.pwconv1 = Conv(c, c * 4, 1)  # Pointwise expand
        self.pwconv2 = Conv(c * 4, c, 1)  # Pointwise compress
        
        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)
        
        # Channel mixing (utilise ConvNoBN pour éviter BatchNorm avec tenseurs 1x1)
        self.channel_mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNoBN(c, c // 4, 1),  # Pas de BatchNorm ici
            nn.SiLU(),
            ConvNoBN(c // 4, c, 1),  # Pas de BatchNorm ici
            nn.Sigmoid()
        )
        
        self.activation = nn.SiLU()
        
    def forward(self, x):
        x1, x2 = x
        
        # Combine inputs
        x_combined = x1 + x2
        shortcut = x_combined
        
        # Depthwise-separable "attention"
        out = self.norm1(x_combined)
        out = self.dwconv(out)
        out = self.activation(out)
        
        # Channel mixing
        channel_weight = self.channel_mixer(out)
        out = out * channel_weight
        
        # Pointwise MLP
        out = self.pwconv1(out)
        out = self.activation(out)
        out = self.pwconv2(out)
        
        # Residual connection
        out = self.norm2(out + shortcut)
        
        return out
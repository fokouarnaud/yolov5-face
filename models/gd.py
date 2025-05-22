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
    Attention-based feature fusion for Low-Stage GD
    """
    def __init__(self, c):
        super(AttentionFusion, self).__init__()
        self.cv1 = Conv(c, c, 1)
        self.cv2 = Conv(c, c, 1)
        self.cv3 = Conv(c, c, 1)
        self.cv_out = Conv(c, c, 3, 1)
        
    def forward(self, x):
        x1, x2 = x
        
        # Calculate attention weights
        attn = torch.sigmoid(self.cv1(x1) + self.cv2(x2))
        out = x1 * attn + x2 * (1 - attn)
        out = self.cv_out(out)
        return out

class TransformerFusion(nn.Module):
    """
    Transformer-like fusion module for High-Stage GD
    Simplified self-attention mechanism
    """
    def __init__(self, c, num_heads=4):
        super(TransformerFusion, self).__init__()
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj = Conv(c, c, 1)
        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)
        self.ffn = nn.Sequential(
            Conv(c, c * 4, 1),
            nn.SiLU(),
            Conv(c * 4, c, 1)
        )
        
    def forward(self, x):
        x1, x2 = x
        
        # Combine inputs
        x_combined = x1 + x2
        shortcut = x_combined
        
        # Self-attention
        B, C, H, W = x_combined.shape
        q = self.q(x_combined).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = self.k(x_combined).reshape(B, self.num_heads, self.head_dim, H * W)
        v = self.v(x_combined).reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.norm1(out + shortcut)
        
        # Feed-forward network
        shortcut = out
        out = self.ffn(out)
        out = self.norm2(out + shortcut)
        
        return out
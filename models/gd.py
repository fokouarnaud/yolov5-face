import torch
import torch.nn as nn
from models.common import Conv

class GDFusion(nn.Module):
    """
    Gather-and-Distribute Fusion module for ADYOLOv5-Face
    Supports both Attention-based fusion and Transformer-like fusion
    """
    def __init__(self, c1, c2, fusion_type='attention'):
        super(GDFusion, self).__init__()
        self.cv1 = Conv(c1[0], c2, 1, 1)  # Adjust first input channel
        self.cv2 = Conv(c1[1], c2, 1, 1)  # Adjust second input channel
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            self.fusion = AttentionFusion(c2)
        elif fusion_type == 'transformer':
            self.fusion = TransformerFusion(c2)
        else:
            self.fusion = nn.Sequential(
                nn.Add(),  # Simple addition as fallback
                Conv(c2, c2, 3, 1)  # Extra convolution for feature refinement
            )
    
    def forward(self, x):
        # x is expected to be a list of two tensors
        x1, x2 = x
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)
        return self.fusion((x1, x2))

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
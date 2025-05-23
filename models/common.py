import math
import numpy as np
import requests
import torch
import torch.nn as nn
import warnings
from PIL import Image, ImageDraw

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list

class DynamicConv2d(nn.Module):
    """Conv2d avec adaptation dynamique des dimensions d'entrée"""
    def __init__(self, c1, c2, k=1, s=1, p=None, bias=True):
        super(DynamicConv2d, self).__init__()
        
        # PyTorch 2+ compatibility: handle tuple/list arguments
        if isinstance(c1, (list, tuple)):
            c1 = c1[0] if len(c1) > 0 else c1
        if isinstance(c2, (list, tuple)):
            c2 = c2[0] if len(c2) > 0 else c2
        if isinstance(k, (list, tuple)):
            k = k[0] if len(k) > 0 else k
        if isinstance(s, (list, tuple)):
            s = s[0] if len(s) > 0 else s
            
        # Convertir les arguments en entiers
        self.c1_expected = int(c1) if not isinstance(c1, str) else 0  # Attendu
        self.c2 = int(c2) if not isinstance(c2, str) else 0
        self.k = int(k) if isinstance(k, (int, float)) else k
        self.s = int(s) if isinstance(s, (int, float)) else s
        self.p = p if p is not None else autopad(self.k)
        self.bias = bias
        
        # Créer les couches dynamiquement dans forward
        self.conv = None
            
    def _create_layer(self, c1_actual):
        # Créer la couche avec les dimensions réelles
        self.conv = nn.Conv2d(c1_actual, self.c2, self.k, self.s, self.p, bias=self.bias)

    def forward(self, x):
        # Détecter les dimensions d'entrée réelles
        c1_actual = x.shape[1]
        
        # Initialiser ou réinitialiser la couche si nécessaire
        if self.conv is None or self.conv.in_channels != c1_actual:
            self._create_layer(c1_actual)
            
        return self.conv(x)# This file contains modules common to various models



def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    # Assurez-vous que c1 et c2 sont des entiers pour le calcul du gcd
    if isinstance(c1, (list, tuple)):
        c1 = c1[0] if len(c1) > 0 else c1
    if isinstance(c2, (list, tuple)):
        c2 = c2[0] if len(c2) > 0 else c2
    c1, c2 = int(c1), int(c2)
    g = math.gcd(c1, c2)
    # Garantir que g est au moins 1
    g = max(1, g)
    return Conv(c1, c2, k, s, g=g, act=act)

class Conv(nn.Module):
    # Standard convolution with dynamic input handling
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        
        # PyTorch 2+ compatibility: handle tuple/list arguments
        if isinstance(c1, (list, tuple)):
            c1 = c1[0] if len(c1) > 0 else c1
        if isinstance(c2, (list, tuple)):
            c2 = c2[0] if len(c2) > 0 else c2
        if isinstance(k, (list, tuple)):
            k = k[0] if len(k) > 0 else k
        if isinstance(s, (list, tuple)):
            s = s[0] if len(s) > 0 else s
        if isinstance(g, (list, tuple)):
            g = g[0] if len(g) > 0 else g
            
        # Convertir les arguments en entiers et s'assurer qu'ils sont positifs
        self.c1_expected = int(c1) if not isinstance(c1, str) else 0  # Attendu
        self.c2 = int(c2) if not isinstance(c2, str) else 0
        self.k = int(k) if isinstance(k, (int, float)) else k
        self.s = int(s) if isinstance(s, (int, float)) else s
        self.p = p
        self.g = int(g) if isinstance(g, (int, float)) else 1
        self.g = max(1, self.g)  # groups doit être au moins 1
        self.act_type = act
        
        # Créer les couches dynamiquement dans forward
        self.conv = None
        self.bn = None
        self.act = None
            
    def _create_layers(self, c1_actual):
        # Créer les couches avec les dimensions réelles
        self.conv = nn.Conv2d(c1_actual, self.c2, self.k, self.s, autopad(self.k, self.p), groups=self.g, bias=False)
        self.bn = nn.BatchNorm2d(self.c2)
        self.act = nn.SiLU() if self.act_type is True else (self.act_type if isinstance(self.act_type, nn.Module) else nn.Identity())

    def forward(self, x):
        # Détecter les dimensions d'entrée réelles
        c1_actual = x.shape[1]
        
        # Initialiser ou réinitialiser les couches si nécessaire
        if self.conv is None or self.conv.in_channels != c1_actual:
            self._create_layers(c1_actual)
            
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        # Détecter les dimensions d'entrée réelles
        c1_actual = x.shape[1]
        
        # Initialiser ou réinitialiser les couches si nécessaire
        if self.conv is None or self.conv.in_channels != c1_actual:
            self._create_layers(c1_actual)
            
        return self.act(self.conv(x))

class StemBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=2, p=None, g=1, act=True):
        super(StemBlock, self).__init__()
        
        # Conversion en entiers si nécessaire
        if isinstance(c1, (list, tuple)):
            c1 = c1[0] if len(c1) > 0 else c1
        if isinstance(c2, (list, tuple)):
            c2 = c2[0] if len(c2) > 0 else c2
            
        self.c1 = int(c1) if not isinstance(c1, str) else 0
        self.c2 = int(c2) if not isinstance(c2, str) else 0
        
        # Initialisation dynamique
        self.stem_1 = None
        self.stem_2a = None
        self.stem_2b = None
        self.stem_2p = None
        self.stem_3 = None
        
        # Paramètres pour créer les couches dynamiquement
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        self.act = act

    def _init_layers(self, c1_actual):
        self.stem_1 = Conv(c1_actual, self.c2, self.k, self.s, self.p, self.g, self.act)
        self.stem_2a = Conv(self.c2, self.c2 // 2, 1, 1, 0)
        self.stem_2b = Conv(self.c2 // 2, self.c2, 3, 2, 1)
        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.stem_3 = Conv(self.c2 * 2, self.c2, 1, 1, 0)

    def forward(self, x):
        # Détecter les dimensions d'entrée réelles
        c1_actual = x.shape[1]
        
        # Initialiser ou réinitialiser les couches si nécessaire
        if self.stem_1 is None:
            self._init_layers(c1_actual)
            
        stem_1_out  = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)
        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))
        return out

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        
        # Conversion en entiers si nécessaire
        if isinstance(c1, (list, tuple)):
            c1 = c1[0] if len(c1) > 0 else c1
        if isinstance(c2, (list, tuple)):
            c2 = c2[0] if len(c2) > 0 else c2
            
        self.c1 = int(c1) if not isinstance(c1, str) else 0
        self.c2 = int(c2) if not isinstance(c2, str) else 0
        self.e = e
        
        # Assurez-vous que g est un entier positif pour PyTorch 2+
        self.g = int(g) if isinstance(g, (int, float)) else 1
        self.g = max(1, self.g)  # g doit être au moins 1
        
        # Initialisation dynamique
        self.cv1 = None
        self.cv2 = None
        self.add = shortcut and self.c1 == self.c2

    def _init_layers(self, c1_actual):
        c_ = int(self.c2 * self.e)  # hidden channels
        self.cv1 = Conv(c1_actual, c_, 1, 1)
        self.cv2 = Conv(c_, self.c2, 3, 1, g=self.g)

    def forward(self, x):
        # Détecter les dimensions d'entrée réelles
        c1_actual = x.shape[1]
        
        # Initialiser ou réinitialiser les couches si nécessaire
        if self.cv1 is None or self.cv1.conv is None or self.cv1.conv.in_channels != c1_actual:
            self._init_layers(c1_actual)
            
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()        
        # Conversion de c1 en entier si c'est une liste/tuple
        if isinstance(c1, (list, tuple)):
            c1 = c1[0] if len(c1) > 0 else c1
        c1 = int(c1)
        c2 = int(c2)
        
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = None
        self.cv2 = None
        self.cv3 = None
        self.m = None
        self.c1 = c1
        self.c2 = c2
        self.c_ = c_
        self.n = n
        self.shortcut = shortcut
        self.g = g
        self.e = e

    def _init_layers(self, c1):
        c_ = self.c_
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, self.c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, self.shortcut, self.g, e=1.0) for _ in range(self.n)])

    def forward(self, x):
        # Initialiser dynamiquement les couches si elles n'existent pas encore
        # ou si la dimension d'entrée a changé
        c1 = x.shape[1]
        if self.cv1 is None or self.cv1.conv.in_channels != c1:
            self._init_layers(c1)
            
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.SiLU(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.SiLU(),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out
    
class BlazeBlock(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=None,stride=1):
        super(BlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride>1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=5,stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1+self.shortcut(x)) if self.use_pool else (branch1+x)
        return self.relu(out)    
  
class DoubleBlazeBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,stride=1):
        super(DoubleBlazeBlock, self).__init__()
        mid_channels = mid_channels or in_channels
        assert stride in [1, 2]
        if stride > 1:
            self.use_pool = True
        else:
            self.use_pool = False

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=stride,padding=2,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
        )

        if self.use_pool:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(kernel_size=stride, stride=stride),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.SiLU(inplace=True)

    def forward(self, x):
        branch1 = self.branch1(x)
        out = (branch1 + self.shortcut(x)) if self.use_pool else (branch1 + x)
        return self.relu(out)
    
    
class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)

class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1 = [], []  # image and inference shapes
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)  # open
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False, render=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save or render:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if pprint:
                print(str)
            if show:
                img.show(f'Image {i}')  # show
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class GatherLayer(nn.Module):
    """Couche de rassemblement pour le mécanisme GD (Gather-and-Distribute)"""
    def __init__(self, out_channels, *args):
        super(GatherLayer, self).__init__()
        # S'assurer que out_channels est un entier
        if isinstance(out_channels, (list, tuple)):
            out_channels = out_channels[0] if len(out_channels) > 0 else out_channels
        self.out_channels = int(out_channels)
        
        # Les couches de convolution seront créées dynamiquement dans forward
        self.cv1 = None
        self.cv2 = None
        
    def forward(self, x):
        # Gérer le cas où x est une liste ou un tuple
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError("GatherLayer s'attend à recevoir une liste ou tuple avec au moins 2 éléments")
            
        # Détecter dynamiquement les dimensions des entrées
        x1, x2 = x[0], x[1]
        in_channels = x1.shape[1]  # Nombre de canaux de l'entrée actuelle
        
        # Créer la couche de convolution adaptée à la dimension de l'entrée
        if self.cv1 is None or self.cv1.conv.in_channels != in_channels:
            self.cv1 = Conv(in_channels, self.out_channels, 1, 1)
            self.cv2 = Conv(self.out_channels + x2.shape[1], self.out_channels, 1, 1)
        
        # x1 est la caractéristique de la couche actuelle
        # x2 est la caractéristique de la couche précédente (upsampled)
        return self.cv2(torch.cat([self.cv1(x1), x2], 1))

class DistributeLayer(nn.Module):
    """Couche de distribution pour le mécanisme GD (Gather-and-Distribute)"""
    def __init__(self, out_channels, *args):
        super(DistributeLayer, self).__init__()
        # S'assurer que out_channels est un entier
        if isinstance(out_channels, (list, tuple)):
            out_channels = out_channels[0] if len(out_channels) > 0 else out_channels
        self.out_channels = int(out_channels)
        
        # Les couches de convolution seront créées dynamiquement dans forward
        self.cv1 = None
        self.cv2 = None
        
    def forward(self, x):
        # Gérer le cas où x est une liste ou un tuple
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError("DistributeLayer s'attend à recevoir une liste ou tuple avec au moins 2 éléments")
            
        # Détecter dynamiquement les dimensions des entrées
        x1, x2 = x[0], x[1]
        in_channels = x1.shape[1]  # Nombre de canaux de l'entrée actuelle
        
        # Créer la couche de convolution adaptée à la dimension de l'entrée
        if self.cv1 is None or self.cv1.conv.in_channels != in_channels:
            self.cv1 = Conv(in_channels, self.out_channels, 1, 1)
            self.cv2 = Conv(self.out_channels + x2.shape[1], self.out_channels, 1, 1)
        
        # x1 est la caractéristique de la couche actuelle (convolved)
        # x2 est la caractéristique de la couche cible
        return self.cv2(torch.cat([self.cv1(x1), x2], 1))



class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

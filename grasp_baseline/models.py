from . import config as cfg
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, resnet34, resnet101, vgg16_bn, vgg11_bn, vgg19_bn, vgg11, vgg16, vgg19
from torchvision.models.mobilenet import mobilenet_v2
from .darknet53 import darknet53

def vgg16_forward(backbone, x):
    return backbone.features(x)

def darknet53_forward(backbone, x):
    return backbone(x)

def resnet50_forward(backbone, x):
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return x

_valid_backbones = {
    'resnet50': resnet50,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet101': resnet101,
    'vgg16_bn' : vgg16_bn,
    'vgg11_bn' : vgg11_bn,
    'vgg19_bn' : vgg19_bn,
    'vgg16' : vgg16,
    'vgg11' : vgg11,
    'vgg19' : vgg19,
    'mobilenetv2' : mobilenet_v2,
    'darknet53' : darknet53,
}

_last_channel_n = {
    'resnet50': 2048,
    'resnet18': 512,
    'resnet34': 512,
    'resnet101': 2048,
    'vgg16': 512,
    'vgg11': 512,
    'vgg19': 512,
    'vgg16_bn': 512,
    'vgg11_bn': 512,
    'vgg19_bn': 512,
    'mobilenetv2' : 1280,
    'darknet53' : 1024,
}

_feature_forward_methods = {
    'resnet50': resnet50_forward,
    'resnet18': resnet50_forward,
    'resnet34': resnet50_forward,
    'resnet101': resnet50_forward,
    'vgg16' : vgg16_forward,
    'vgg11' : vgg16_forward,
    'vgg19' : vgg16_forward,
    'vgg16_bn' : vgg16_forward,
    'vgg11_bn' : vgg16_forward,
    'vgg19_bn' : vgg16_forward,
    'mobilenetv2': vgg16_forward,
    'darknet53' : darknet53_forward,
}

def grasp_model(backbone_name):
    assert backbone_name in _valid_backbones
    backbone = _valid_backbones[backbone_name](pretrained=True)
    if hasattr(backbone, 'classifier'):
        del backbone.classifier
    if hasattr(backbone, 'avgpool'):
        del backbone.avgpool
    if hasattr(backbone, 'fc'):
        del backbone.fc
    if hasattr(backbone, 'fc1'):
        del backbone.fc1
    if hasattr(backbone, 'fc2'):
        del backbone.fc2
    feature_forward = _feature_forward_methods[backbone_name]
    last_channel_n = _last_channel_n[backbone_name]
    return GraspModel(backbone, feature_forward, last_channel_n)

class GraspModel(nn.Module):
    def __init__(self, backbone, feature_forward, last_channel_n):
        super(GraspModel, self).__init__()
        self.backbone = backbone
        self.feature_forward = feature_forward
        self.conf_layer = nn.Conv2d(last_channel_n, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.xy_layer = nn.Conv2d(last_channel_n, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.wh_layer = nn.Conv2d(last_channel_n, 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.cossine_layer = nn.Conv2d(last_channel_n, 2, kernel_size=1, stride=1, padding=0, bias=True)
        #self.angle_linear = nn.Conv2d(last_channel_n, cfg.n_orientations, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.feature_forward(self.backbone, x)
        conf = self.conf_layer(x)
        xy = torch.sigmoid(self.xy_layer(x))
        wh = torch.exp(self.wh_layer(x)) # YOLOv2
        tha = self.cossine_layer(x)
        #tha = self.angle_linear(x)
        #tha = torch.sigmoid(tha)
        tha = tha / torch.norm(tha, p=2, dim=1, keepdim=True) # normalized rotation vector
        x = torch.cat((conf, xy, wh, tha), 1) # (b, c, h, w)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    #for backbone in ['vgg11', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg16_bn', 'vgg19_bn', 'mobilenetv2', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'darknet53']:
    for backbone in ['darknet53',]:
        grasp_model = grasp_model(backbone)
        print(grasp_model)
        y = grasp_model(torch.randn((6,3,cfg.input_size,cfg.input_size)))
        z = y.mean()
        z.backward()
        print(y.size(), count_parameters(grasp_model))
        del grasp_model

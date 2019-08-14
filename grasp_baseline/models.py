from . import config as cfg
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18, resnet34, resnet101, vgg16_bn, vgg11_bn, vgg19_bn, vgg11, vgg16, vgg19
from torchvision.models.mobilenet import mobilenet_v2

class GraspModel(nn.Module):
    def __init__(self, backbone='vgg19', with_fc=False):
        super(GraspModel, self).__init__()
        self.valid_backbones = {
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
        }
        assert backbone in self.valid_backbones
        self.last_channel_n = {
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
        }
        self.backbone = self.valid_backbones[backbone](pretrained=True)
        if hasattr(self.backbone, 'classifier'):
            del self.backbone.classifier
        if hasattr(self.backbone, 'avgpool'):
            del self.backbone.avgpool
        if hasattr(self.backbone, 'fc'):
            del self.backbone.fc
        if hasattr(self.backbone, 'fc1'):
            del self.backbone.fc1
        if hasattr(self.backbone, 'fc2'):
            del self.backbone.fc2
        self.feature_forward_methods = {
            'resnet50': self.resnet50_forward,
            'resnet18': self.resnet50_forward,
            'resnet34': self.resnet50_forward,
            'resnet101': self.resnet50_forward,
            'vgg16' : self.vgg16_forward,
            'vgg11' : self.vgg16_forward,
            'vgg19' : self.vgg16_forward,
            'vgg16_bn' : self.vgg16_forward,
            'vgg11_bn' : self.vgg16_forward,
            'vgg19_bn' : self.vgg16_forward,
            'mobilenetv2': self.vgg16_forward
        }
        self.feature_forward = self.feature_forward_methods[backbone]
        self.with_fc = with_fc
        if self.with_fc:
            self.fc1 = nn.Linear(cfg.grid_size*cfg.grid_size*self.last_channel_n[backbone], 512)
            self.fc2 = nn.Linear(512, 512*cfg.grid_size*cfg.grid_size)
            self.conf_layer = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.xy_layer = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.wh_layer = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
            #self.cossine_layer = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.angle_linear = nn.Conv2d(512, cfg.n_orientations, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.conf_layer = nn.Conv2d(self.last_channel_n[backbone], 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.xy_layer = nn.Conv2d(self.last_channel_n[backbone], 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.wh_layer = nn.Conv2d(self.last_channel_n[backbone], 2, kernel_size=1, stride=1, padding=0, bias=True)
            #self.cossine_layer = nn.Conv2d(self.last_channel_n[backbone], 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.angle_linear = nn.Conv2d(self.last_channel_n[backbone], cfg.n_orientations, kernel_size=1, stride=1, padding=0, bias=True)
        self.forward = self.with_fc_forward if self.with_fc else self.without_fc_forward

    def with_fc_forward(self, x):
        x = self.feature_forward(x)
        x = x.view(x.size(0), -1) # (b, c*h*w)
        x = F.leaky_relu(self.fc1(x), 0.1) # (b, 512)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.fc2(x), 0.1) # (b, 512*7*7)
        x = x.view(x.size(0), 512, cfg.grid_size, cfg.grid_size) # (b, 512, 7 ,7)
        conf = self.conf_layer(x)
        xy = torch.sigmoid(self.xy_layer(x))
        wh = torch.exp(self.wh_layer(x)) # YOLOv2
        #tha = self.cossine_layer(x)
        tha = self.angle_linear(x)
        tha = torch.sigmoid(tha)
        x = torch.cat((conf, xy, wh, tha), 1) # (b, c, h, w)
        return x

    def without_fc_forward(self, x):
        x = self.feature_forward(x)
        conf = self.conf_layer(x)
        xy = torch.sigmoid(self.xy_layer(x))
        wh = torch.exp(self.wh_layer(x)) # YOLOv2
        #tha = self.cossine_layer(x)
        tha = self.angle_linear(x)
        tha = torch.sigmoid(tha)
        x = torch.cat((conf, xy, wh, tha), 1) # (b, c, h, w)
        return x

    def vgg16_forward(self, x):
        return self.backbone.features(x)

    def resnet50_forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    for backbone in ['vgg11', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg16_bn', 'vgg19_bn', 'mobilenetv2', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
        for mode in [True, False]:
            grasp_model = GraspModel(backbone=backbone, with_fc=mode)
            print(grasp_model)
            y = grasp_model(torch.randn((6,3,cfg.input_size,cfg.input_size)))
            z = y.mean()
            z.backward()
            print(y.size(), count_parameters(grasp_model))
            del grasp_model

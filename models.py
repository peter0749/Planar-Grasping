import config as cfg
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet50, vgg16_bn

class GraspModel(nn.Module):
    def __init__(self, backbone='vgg16', with_fc=False):
        super(GraspModel, self).__init__()
        self.valid_backbones = {
            'resnet50': resnet50,
            'vgg16' : vgg16_bn,
        }
        assert backbone in self.valid_backbones
        self.last_channel_n = {
            'resnet50': 2048,
            'vgg16': 512,
        }
        self.backbone = self.valid_backbones[backbone](pretrained=True)
        if hasattr(self.backbone, 'classifier'):
            del self.backbone.classifier
        if backbone=='resnet50':
            if hasattr(self.backbone, 'avgpool'):
                del self.backbone.avgpool
            if hasattr(self.backbone, 'fc'):
                del self.backbone.fc
        self.feature_forward_methods = {
            'resnet50': self.resnet50_forward,
            'vgg16' : self.vgg16_forward,
        }
        self.feature_forward = self.feature_forward_methods[backbone]
        self.with_fc = with_fc
        if self.with_fc:
            self.fc1 = nn.Linear(cfg.grid_size*cfg.grid_size*self.last_channel_n[backbone], 512)
            self.fc2 = nn.Linear(512, 512*cfg.grid_size*cfg.grid_size)
            self.conf_layer = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.xy_layer = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.wh_layer = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.cossine_layer = nn.Conv2d(512, 2, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.conf_layer = nn.Conv2d(self.last_channel_n[backbone], 1, kernel_size=1, stride=1, padding=0, bias=True)
            self.xy_layer = nn.Conv2d(self.last_channel_n[backbone], 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.wh_layer = nn.Conv2d(self.last_channel_n[backbone], 2, kernel_size=1, stride=1, padding=0, bias=True)
            self.cossine_layer = nn.Conv2d(self.last_channel_n[backbone], 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.forward = self.with_fc_forward if self.with_fc else self.without_fc_forward

    def with_fc_forward(self, x):
        x = self.feature_forward(x)
        x = x.view(x.size(0), -1) # (b, c*h*w)
        x = F.leaky_relu(self.fc1(x), 0.1) # (b, 512)
        x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.fc2(x), 0.1) # (b, 512*7*7)
        x = x.view(x.size(0), 512, cfg.grid_size, cfg.grid_size) # (b, 512, 7 ,7)
        conf = torch.sigmoid(self.conf_layer(x))
        xy = torch.sigmoid(self.xy_layer(x))
        wh = self.wh_layer(x)
        tha = self.cossine_layer(x) # 0~2pi
        x = torch.cat((conf, xy, wh, tha), 1) # (b, c, h, w)
        return x

    def without_fc_forward(self, x):
        x = self.feature_forward(x)
        conf = torch.sigmoid(self.conf_layer(x))
        xy = torch.sigmoid(self.xy_layer(x))
        wh = self.wh_layer(x)
        tha = self.cossine_layer(x) # 0~2pi
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
    for backbone in ['vgg16', 'resnet50']:
        for mode in [True, False]:
            grasp_model = GraspModel(backbone=backbone, with_fc=mode)
            print(grasp_model)
            y = grasp_model(torch.randn((6,3,224,224)))
            z = y.mean()
            z.backward()
            print(y.size(), count_parameters(grasp_model))
            del grasp_model

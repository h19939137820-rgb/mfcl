import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from our_model.danet import DANetHead
from our_model.maspp import MASPP
from our_model.sedge import SEdge
from our_model.inceptionNeXt import inceptionnext_base_384


class self_Module(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(self_Module, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.masppback = MASPP(in_channels, aspp_dilate)
        self.dattion = DANetHead(256, 256, nn.BatchNorm2d)
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=1)
        )
        self.atten = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.getedge = SEdge()
        self.cat_last = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1)
        )

        self._init_weight()

    def forward(self, feature, low_level, more_features):  # edge 8 2 64 64
        edge = self.getedge(more_features)
        low_level_feature = self.project(low_level)
        out = feature
        out = self.masppback(out)
        out_da = self.dattion(out)
        out = self.atten(out + out_da)
        out = self.cat_last(out)
        out = F.interpolate(out, size=low_level_feature.shape[2:], mode='bilinear',
                            align_corners=False)
        return self.classifier(torch.cat([low_level_feature, out], dim=1)), self.decoder(
            torch.cat([low_level_feature, out], dim=1)), edge

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def loadweights(model):
    path = "please input your Backbone .pth"
    print('Load weights {}.'.format(path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location=device)
    print(pretrained_dict.keys())
    print(model_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
    # 更新模型参数
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(
        f"Model weights loaded successfully with partial matching.pretrained_dict:{len(pretrained_dict.keys())},model_dict:{len(model_dict.keys())}")


class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        features, low_level = x[-1], x[0]
        x, feat, edge, onlyCl, onlyCE = self.classifier(features, low_level, x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x, feat, edge


def get_MFCL(num_classes):
    aspp_dilate = [12, 24, 36]
    backbone = inceptionnext_base_384()
    loadweights(backbone)
    inplanes = 1024
    low_level_planes = 128
    classifier = self_Module(inplanes, low_level_planes, num_classes, aspp_dilate)
    model = Net(backbone, classifier)
    return model


class Net(_SimpleSegmentationModel):
    pass

if __name__ == '__main__':
    device = "cuda"
    in_data = torch.randint(0, 255, (1, 256, 384, 384), dtype=torch.float32).to(device)
    model = get_MFCL(2).to(device)
    x,con,edge = model(in_data)
    print(x.shape)
    print(con.shape)
    print(edge.shape)
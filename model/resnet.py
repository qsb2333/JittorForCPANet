import jittor as jt
from jittor import nn


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet50s-a75c83cf.zip',
    'resnet101': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet101s-03a0f310.zip',
    'resnet152': 'https://s3.us-west-1.wasabisys.com/encoding/models/resnet152s-36670e8b.zip'
}

#res18 res34
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,stride=1,dilation=1,downsample=None,norm_layer=None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=dilation,dilation=dilation,bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.Relu()
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,dilation=1,bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out +=residual
        out = self.relu(out)

        return out

#res50 res101 res152
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inplanes,planes,stride=1,dilation=1,downsample=None,norm_layer=None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=dilation,dilation=dilation,bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes,planes*4,kernel_size=1,bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.Relu()
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out +=residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,block, layers,num_classes=1000,deep_base=True,norm_layer=nn.BatchNorm2d):
        super(ResNet,self).__init__()
        self.inplanes = 128 if deep_base else 64
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
                norm_layer(64),
                nn.Relu(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False),
                norm_layer(64),
                nn.Relu(),
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.Relu()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0],norm_layer=norm_layer)
        self.layer2 = self._make_layer(block,128,layers[1],stride=2,norm_layer=norm_layer) #尺寸变化
        self.layer3 = self._make_layer(block,256,layers[2],stride=2,norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)

        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc = nn.Linear(512 * block.expansion,num_classes)

        # He初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def _make_layer(self,block, planes, blocks, stride=1,dilation=1,norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes * block.expansion,kernel_size=1, stride=stride,bias=False),
                norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=1,downsample=downsample,norm_layer=norm_layer))

        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes, planes,dilation=dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False,**kwargs):
    model = ResNet(BasicBlock,[2,2,2,2],deep_base=False,**kwargs)
    if pretrained:
        model_path = './initmodel/resnet18_v2.pth'
        model.load_state_dict(jt.load(model_path))
    return model

def resnet34(pretrained=False,**kwargs):
    model = ResNet(BasicBlock,[3,4,6,3],deep_base=False,**kwargs)
    if pretrained:
        model_path = './initmodel/resnet34_v2.pth'
        model.load_state_dict(jt.load(model_path))
    return model

def resnet50(pretrained=False,**kwargs):
    model = ResNet(Bottleneck,[3, 4, 6, 3],deep_base=True)
    if pretrained:
        model_path = '../initmodel/resnet50_v2.pth'
        model.load_state_dict(jt.load(model_path))
    return model

def resnet101(pretrained=False,**kwargs):
    model = ResNet(Bottleneck,[3,4,23,3],deep_base=True)
    if pretrained:
        model_path = './initmodel/resnet101_v2.pth'
        model.load_state_dict(jt.load(model_path))
    return model

def resnet152(pretrained=False,**kwargs):
    model = ResNet(Bottleneck,[3,4,36,3],deep_base=True)
    if pretrained:
        model_path = './initmodel/resnet152_v2.pth'
        model.load_state_dict(jt.load(model_path))
    return model

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    resnet = resnet50(pretrained=True)
    modifyName = 'conv2'#1 表示resnet34 与 18  2 表示resnet50 与 101
    resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4 = resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4
    # 空洞卷积
    for name, module in resnet.layer3.named_modules():
        if modifyName in name:
            module.dilation = (2,2)
            module.padding = (2,2)
            module.stride = (1,1)
        elif 'downsample.0' in name:
            module.stride = (1,1)


    for name, module in resnet.layer4.named_modules():
        if modifyName in name:
            module.dilation = (4, 4)
            module.padding = (4, 4)
            module.stride = (1, 1)
        elif 'downsample.0' in name:
            module.stride = (1, 1)
    print(resnet)
    # # 创建输入张量
    # x = jt.randn(4, 3, 200, 200)
    # print("Input shape:", x.shape)
    #
    # # 提取特征
    # query_feat_0 = resnet.layer0(x)
    # print("query_feat_0 shape:", query_feat_0.shape)
    #
    # query_feat_1 = resnet.layer1(query_feat_0)
    # print("query_feat_1 shape:", query_feat_1.shape)
    #
    # query_feat_2 = resnet.layer2(query_feat_1)
    # print("query_feat_2 shape:", query_feat_2.shape)
    #
    # query_feat_3 = resnet.layer3(query_feat_2)
    # print("query_feat_3 shape:", query_feat_3.shape)
    #
    # query_feat_4 = resnet.layer4(query_feat_3)
    # print("query_feat_4 shape:", query_feat_4.shape)
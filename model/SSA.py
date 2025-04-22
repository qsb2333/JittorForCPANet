import jittor as jt
from jittor import nn

from model.CBAM import Cbam

#空间压缩注意力
class SSA(nn.Module):
    def __init__(self):
        super(SSA,self).__init__()
        channels = 256

        #第一级减半下采样
        self.Done1 = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=(2,2),stride=(2,2),padding=0,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.1)
        )
        #第二级减半下采样
        self.Done2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(2, 2), stride=(2, 2), padding=0,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.1)
        )

        #拼接融合降维和特征提取
        self.convCat = nn.Sequential(
            nn.Conv2d(in_channels=channels*3,out_channels=channels,kernel_size=(1,1),stride=(1,1),padding=0,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.1),
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=(3,3),stride=(1,1),padding=1,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.1)
        )

        #残差卷积
        self.convRes = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=(3,3),stride=(1,1),padding=1,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.1)
        )

        #卷积分类头
        self.clsHead = nn.Sequential(
            nn.Conv2d(in_channels=channels,out_channels=2,kernel_size=(1,1))
        )

        #混合注意力机制
        self.Cbam = Cbam(256)

        self.init_weight()

    def execute(self, x):
        #下采样、上采样 + cat
        x1 = self.Done1(x)
        x1_up = nn.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=True)
        x2 = self.Done2(x1)
        x2_up = nn.interpolate(x2,scale_factor=4,mode='bilinear',align_corners=True)
        x_cat = jt.concat([x,x1_up,x2_up],dim=1)

        x_fusion = self.convCat(x_cat) #[b,256,200,200]
        x_res = self.convRes(x_fusion)
        x_att=self.Cbam(x_fusion)
        output = self.clsHead(x_att + x_res)

        return output

    # He初始化
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


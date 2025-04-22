import jittor as jt
from jittor import nn

#通道注意力
class channeAtt(nn.Module):
    def __init__(self, channel=256, ratio=16):
        super(channeAtt, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        b, c, h, w = x.size()
        maxpool = self.max_pool(x).view([b, c])
        avgpool = self.avg_pool(x).view([b, c])
        maxfc = self.fc(maxpool)
        avgfc = self.fc(avgpool)

        out = maxfc + avgfc
        out = self.sigmoid(out).view([b, c, 1, 1])

        return out * x


class spacialAtt(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacialAtt, self).__init__()
        padding = 7 // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        maxpool= x.max(dim=1, keepdim=True)  # 通道最大池化
        meanpool = x.mean(dim=1, keepdim=True)  # 通道平均池化
        pool_out = jt.concat([maxpool, meanpool], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return out * x


class Cbam(nn.Module):
    def __init__(self, channel=256, ratio=16, kernel_size=7):
        super(Cbam, self).__init__()
        self.channeAtt = channeAtt(channel, ratio)
        self.spacialAtt = spacialAtt(kernel_size)

    def execute(self, x):
        x = self.channeAtt(x)
        x = self.spacialAtt(x)
        return x

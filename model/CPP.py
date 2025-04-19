import jittor as jt
from jittor import nn


#非局部注意力机制
class CPP(nn.Module):
    def __init__(self, in_channel, sub_sample=True):
        super(CPP,self).__init__()
        self.in_channel = in_channel
        self.sub_sample = sub_sample
        self.med_channel = in_channel // 2

        #Val的映射
        self.V = nn.Conv2d(in_channels=self.in_channel, out_channels=self.med_channel,kernel_size=(1,1),stride=(1,1),padding=0)
        # query的映射
        self.Q = nn.Conv2d(in_channels=self.in_channel, out_channels=self.med_channel, kernel_size=(1, 1),stride=(1, 1), padding=0)
        # key的映射
        self.K = nn.Conv2d(in_channels=self.in_channel, out_channels=self.med_channel, kernel_size=(1, 1),stride=(1, 1), padding=0)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        #下采样对
        if sub_sample:
            self.V = nn.Sequential(self.V,nn.MaxPool2d(kernel_size=(2,2)))
            self.K = nn.Sequential(self.K,nn.MaxPool2d(kernel_size=(2,2)))

        #恢复
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.med_channel,out_channels=self.in_channel,kernel_size=(1,1),stride=(1,1),padding=0),
            nn.BatchNorm2d(self.in_channel)
        )
        nn.init.constant_(self.W[1].weight,0)
        nn.init.constant_(self.W[1].bias, 0)


    def execute(self, x):
        '''
        :param x: (b,c,h,w)
        :return: (b,c,1,1)
        '''
        b, c, h, w = x.size()
        #V矩阵（b,c/2,h*w/4)
        x_v = self.V(x).view(b,self.med_channel,-1)
        x_v = x_v.permute(0, 2, 1)
        #Q矩阵（b,c/2,h*w）
        x_q = self.Q(x).view(b,self.med_channel,-1)
        x_q = x_q.permute(0, 2, 1)
        # K矩阵（b,c/2,h*w/4）
        x_k = self.K(x)
        x_k = x_k.view(b, self.med_channel, -1)
        #注意力分数（b,h*w,h*w/4）
        A = nn.softmax(nn.matmul(x_q,x_k),dim=-1)
        out = nn.matmul(A,x_v) #（b,h*w,c/2）
        out = out.permute(0,2,1).view(b,self.med_channel,h,w)
        W_out = self.W(out)
        output = self.avg_pool(W_out + x)
        return output

if __name__ == "__main__":
    model = CPP(256)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params: %.4fM" % (total_params / 1e6))
    x = jt.ones((4, 256, 25, 25))
    out = model(x)
    print("Output shape:", out.shape)



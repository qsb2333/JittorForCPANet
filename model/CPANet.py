import jittor as jt
from jittor import nn, reshape


import model.resnet as models
import model.vgg as vgg_models
from model.CPP import CPP
from model.SSA import SSA


def get_vgg16_layer(model):
    '''
    layer0	0 - 6	Conv1_1, Conv1_2 + MaxPool1	64(包含relu等层)
    layer1	7 - 13	Conv2_1, Conv2_2 + MaxPool2	128
    layer2	14 - 23	Conv3_1~3 + MaxPool3	256
    layer3	24 - 33	Conv4_1~3 + MaxPool4	512
    layer4	34 - 42	Conv5_1~3 + MaxPool5	512
    '''
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4

class cpanet(nn.Module):
    def __init__(self,layers=50, classes=2,criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True,shot=1,vgg=False):
        super(cpanet,self).__init__()
        self.criterion = criterion
        self.shot = shot
        self.vgg =vgg

        if self.vgg:
            print('>>>>>>>>> Using VGG_16 bn <<<<<<<<<')
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            self.layer0,self.layer1,self.layer2,self.layer3,self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('>>>>>>>>> Using ResNet {} <<<<<<<<<'.format(layers))
            if layers == 50:
                resnet = models.resnet50(8)
            elif layers ==101:
                resnet = models.resnet101(pretrained=pretrained)
            elif layers ==152:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1,resnet.maxpool)
            # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1,
            #                             resnet.conv2, resnet.bn2, resnet.relu2,
            #                             resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1,  self.layer2,  self.layer3,  self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            #空洞卷积参置数设
            for n,m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation,m.padding,m.stride = (2,2),(2,2),(1,1)
                elif 'downsample.0' in n:
                    m.stride = (1,1)

            for n,m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation,m.padding,m.stride = (4,4),(4,4),(1,1)
                elif 'downsample.0' in n:
                    m.stride = (1,1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        #辅助loss的分类头
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim,reduce_dim,kernel_size=3,padding=1,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2),
            nn.Conv2d(reduce_dim,reduce_dim,kernel_size=3,padding=1,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2),
            nn.Conv2d(reduce_dim,2,kernel_size=3,padding=1,bias=False)
        )

        #query图像降维
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim,reduce_dim,kernel_size=1,padding=0,bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2)
        )
        #support图像降维
        self.down_support = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2)
        )
        #Fq+Fs拼接
        self.conv_Fsq = nn.Sequential(
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2)
        )

        #query的mask精炼
        self.conv_queryMask = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2)
        )

        # support的mask精炼 即SA模块
        self.conv_supportMask = nn.Sequential(
            nn.Conv2d(reduce_dim * 3, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.Relu(),
            nn.Dropout(p=0.2)
        )

        #空间压缩注意力
        self.SSA = SSA()
        #跨位置代理
        self.CPP = CPP(reduce_dim)

    def execute(self, x, s_x, s_y,y=None):
        #(4,3,200,200)
        x_size = x.size()
        h = int(x_size[-1])
        w = int(x_size[-2])

        with jt.no_grad():
            #提取query特征
            query_feat0 = self.layer0(x) #(4,128,50,50)
            query_feat1 = self.layer1(query_feat0) #（4，256，50，50）
            query_feat2 = self.layer2(query_feat1) # （4，512，25，25）
            query_feat3 = self.layer3(query_feat2) #（4，1024，25，25）

            if self.vgg:
                #需要上采样
                query_feat2 = nn.interpolate(query_feat2,size=(query_feat3.size(2),query_feat3.size(3)),
                                             mode='bilinear',align_corners=True)

        query_feat = jt.concat([query_feat3,query_feat2],dim=1)
        query_feat = self.down_query(query_feat)

        gt_list = []
        gt_down_list = []
        support_feat_list=[]
        support_prototype_list = []

        for i in range(self.shot):
            support_gt = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            gt_list.append(support_gt)

            with jt.no_grad():
                # 提取support特征
                support_feat0 = self.layer0(s_x[:,i,:,:,:])  # (4,128,50,50)
                support_feat1 = self.layer1(support_feat0)  # （4，256，50，50）
                support_feat2 = self.layer2(support_feat1)  # （4，512，25，25）
                support_feat3 = self.layer3(support_feat2)  # （4，1024，25，25）

                if self.vgg:
                    # 需要上采样
                    support_feat2 = nn.interpolate(support_feat2, size=(support_feat3.size(2), support_feat3.size(3)),
                                                 mode='bilinear', align_corners=True)
            #maks进行下采样，为了masking操作
            support_gt_down = nn.interpolate(support_gt,size=(support_feat3.size(2),support_feat3.size(3)),
                                             mode='bilinear', align_corners=True)
            gt_down_list.append(support_gt_down)

            support_feat = jt.concat([support_feat3,support_feat2],dim = 1)
            support_feat = self.down_support(support_feat)# （4，256，25，25）
            support_feat_list.append(support_feat)

            support_feat_mask = support_feat * support_gt_down
            #提取原型
            support_prototype = self.CPP(support_feat_mask)
            support_prototype_list.append(support_prototype)

        support_temp = jt.zeros_like(support_prototype_list[0])
        for i in range(self.shot):
            support_temp += support_prototype_list[i]
        support_prototype_mean = support_temp / len(support_temp)
        #expand操作
        support_prototype_mean = support_prototype_mean.expand(query_feat.shape[0],256,query_feat.shape[-2],query_feat.shape[-1])

        #获取Fsq
        Fsq = jt.concat([support_prototype_mean,query_feat],dim = 1)
        Fsq = self.conv_Fsq(Fsq)

        query_pred_mask = self.conv_queryMask(Fsq)
        #恢复原尺寸
        query_pred_mask = nn.interpolate(query_pred_mask,size=(h,w),mode='bilinear',align_corners=True)
        #SSA
        query_pred_mask = self.SSA(query_pred_mask) #[4,2,200,200,]

        query_pred_mask_save = jt.argmax(query_pred_mask[0].permute(1,2,0),dim=-1)[0].detach().numpy()
        query_pred_mask_save[query_pred_mask_save != 0] = 255
        query_pred_mask_save[query_pred_mask_save==0] = 0

        #辅助loss
        support_pred_mask_list = []
        if self.is_training():
            for i in range(self.shot):
                support_prototype_i = support_prototype_list[i]
                support_feat_i = support_feat_list[i]
                #expand
                support_prototype_i = support_prototype_i.expand(support_feat_i.shape[0],256,support_feat_i.shape[-2],support_feat_i.shape[-1])
                #两倍pc拼接
                support_fusion = jt.concat([support_feat_i,support_prototype_i,support_prototype_i],dim=1)
                support_pred_mask = self.conv_supportMask(support_fusion)

                support_pred_mask = nn.interpolate(support_pred_mask, size=(h, w), mode='bilinear', align_corners=True)

                support_pred_mask = self.cls(support_pred_mask)
                support_pred_mask_list.append(support_pred_mask)

        #辅助loss的权重
        k = 0.4
        if self.is_training():
            loss = 0.
            for i in range(self.shot):
                suppot_loss = self.criterion(support_pred_mask_list[i],gt_list[i].squeeze(1).long())
                loss +=suppot_loss
            aux_loss = loss/self.shot
            main_loss = self.criterion(query_pred_mask,y.long())
            return jt.argmax(query_pred_mask,1)[0], main_loss + k * aux_loss
        else:
            return query_pred_mask,query_pred_mask_save


if __name__ == '__main__':
    jt.flags.use_cuda = 1
    model = cpanet(shot=1)
    model.train()
    x = jt.rand(1, 3, 200, 200)
    s_x = jt.rand(1, 1, 3, 200, 200)
    s_y = jt.ones((1, 1,200,200))
    y = jt.ones((1, 1,200,200))
    pred, loss = model(x, s_x, s_y,y)
    print("pred shape:", pred.shape, "loss:", loss.item())

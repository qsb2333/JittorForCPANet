import argparse
import logging
import os
import random
import time

import cv2
import numpy as np

import jittor as jt
from jittor import dataset
from jittor import nn
from torch.utils.tensorboard import SummaryWriter
from model.CPANet import cpanet
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnion,load_cfg_from_yamlFile
from util import dataset, transform

jt.flags.use_cuda = 1
cv2.ocl.setUseOpenCL(False)   # 禁用 GPU 加速（OpenCL）
cv2.setNumThreads(0)          # 禁用 OpenCV 多线程

def get_parser():
    parser = argparse.ArgumentParser(description='Few-Shot Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/SSD/fold1_vgg16.yaml', help='config file')
    args = parser.parse_args()
    cfg = load_cfg_from_yamlFile(args.config)
    return cfg

def DataLoader(dataset, *args, **kargs):
    return dataset.set_attrs(*args, **kargs)

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    #定义日志格式
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger




def main():
    global args
    args = get_parser()

    if args.manual_seed is not None:
        jt.set_global_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = cpanet(layers=args.layers, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255),
                   pretrained=True,shot=args.shot,vgg=args.vgg)

    #冻结resnet等的骨干网络
    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False

    optimizer = jt.optim.SGD(
        [
            {'params': model.down_query.parameters()},
            {'params': model.down_support.parameters()},
            {'params': model.CPP.parameters()},
            {'params': model.cls.parameters()},
            {'params': model.conv_Fsq.parameters()},
            {'params': model.conv_queryMask.parameters()},
            {'params': model.SSA.parameters()},
            {'params': model.conv_supportMask.parameters()},
        ],
        lr=args.base_lr,momentum=args.momentum,weight_decay=args.weight_decay
    )
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    # \033[1;36m 淡青色字体的 ANSI 控制符
    logger.info("\033[1;36m >>>>>>Creating model ...\033[0m")
    logger.info("\033[1;36m >>>>>>Classes: {}\033[0m".format(args.classes))
    logger.info(model)
    print(args)

    value_scale = 255
    #imageNet图像的均值方差
    mean = [0.485,0.456,0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([args.scale_min,args.scale_max]),
        transform.RandRotate([args.rotate_min,args.rotate_max],padding=mean,ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h,args.train_w],crop_type='rand',padding=mean,ignore_label=args.padding_label),
        transform.ToVar(),
        transform.Normalize(mean=mean,std=std)
    ])

    train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,data_list=args.train_list,
                                 transform=train_transform,mode='train')
    train_loader = jt.dataset.DataLoader(dataset=train_data,batch_size=args.batch_size,shuffle=True,
                              drop_last=True,num_workers=args.workers)

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToVar(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToVar(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split=args.split,shot=args.shot,data_root=args.data_root,data_list=args.val_list,
                                  transform=val_transform,mode='val')
    val_loader = jt.dataset.DataLoader(val_data,batch_size=args.batch_size_val,shuffle=False,num_workers=args.workers)

    max_class_iou = 0.
    max_fb_iou = 0
    best_epoch = 0

    filename = 'CPANet.pth'

    for epoch in range(args.start_epoch, args.epochs):

        if args.fix_random_seed_val:
            jt.set_global_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed+ epoch)
            random.seed(args.manual_seed+ epoch)

        epoch_log = epoch + 1
        #训练
        loss_train, mBFIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mBFIoU_train', mBFIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        #验证
        loss_val, mBFIoU, mAcc, allAcc, class_miou = validate(val_loader, model, criterion)
        writer.add_scalar('loss_val', loss_val, epoch_log)
        writer.add_scalar('mBFIoU_val', mBFIoU, epoch_log)
        writer.add_scalar('mAcc_val', mAcc, epoch_log)
        writer.add_scalar('class_miou_val', class_miou, epoch_log)
        writer.add_scalar('allAcc_val', allAcc, epoch_log)
        if class_miou > max_class_iou:
            max_class_iou = class_miou
            best_epoch = epoch
            if os.path.exists(filename):
                os.remove(filename)
            filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_class_iou) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            jt.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},filename)

        if mBFIoU > max_fb_iou:
            max_fb_iou = mBFIoU
        logger.info('Best Epoch {:.1f}, Best IOU {:.4f} Best FB-IoU {:4F}'.format(best_epoch, max_class_iou, max_fb_iou))

    filename = args.save_path + '/final.pth'
    logger.info('Saving Last checkpoint to: ' + filename)
    jt.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

def train(train_loader, model, optimizer, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Train <<<<<<<<<<<<<<<<<<')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)

    for i ,(input, target, s_input, s_mask, subcls,ori_label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        #总迭代batch
        current_iter = epoch * len(train_loader) + i + 1

        #poly策略
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer,args.base_lr,current_iter,max_iter,power=args.power,
                               warmup=args.warmup,warmup_step=len(train_loader) // 2)

        output,loss = model(s_x=s_input, s_y=s_mask, x=input, y=target)
        loss = jt.mean(loss)
        optimizer.step(loss)

        n = input.size(0)
        intersection, union, target = intersectionAndUnion(output,target,args.classes,args.ignore_label)
        intersection, union, target = intersection.numpy(), union.numpy(), target.numpy()
        intersection_meter.update(intersection),union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(),n)
        batch_time.update(time.time() - end)
        end = time.time()
        # 估算剩余训练时间
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % 10 == 0:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
        writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
        writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
        writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum  + 1e-10)
    #BFIOU
    mBFIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    logger.info(
        'Train result at epoch [{}/{}]: mBFIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mBFIoU,
                                                                                       mAcc, allAcc))
    for i in range(args.classes):
        logger.info('BF_Class_{} Result: iou： {:.4f} - accuracy： {:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Train <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mBFIoU, mAcc, allAcc



def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if args.manual_seed is not None and args.fix_random_seed_val:
        jt.set_global_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)

    split_gap = 4
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    model.eval()
    end = time.time()
    test_num = len(val_loader)
    iter_num = 0
    for i, (input,target,s_input,s_mask,subcls,ori_label) in enumerate(val_loader):
        iter_num += 1

        data_time.update(time.time() - end)
        output,_ = model(s_x=s_input,s_y=s_mask,x=input,y=target)
        if args.ori_resize:
            longerside = max(ori_label.size(1), ori_label.size(2))
            backmask = jt.ones((ori_label.size(0), longerside, longerside)) * 255
            backmask[:, :ori_label.size(1), :ori_label.size(2)] = ori_label
            target = backmask.clone().long()

        output = nn.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)

        loss = criterion(output, target)
        output = jt.argmax(output,1)[0]

        intersection, union, target = intersectionAndUnion(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.numpy(), union.numpy(), target.numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        #计算具体缺陷类别的交并集信息 查询集subcls只有1个类别
        subcls = subcls[0].numpy()[0]
        class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
        class_union_meter[(subcls - 1) % split_gap] += union[1]

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(),input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if ((i + 1) % 10 == 0):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(iter_num * args.batch_size_val, test_num,
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
    #这里的class指前景和背景
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mBFIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    #具体的缺陷类别
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou +=class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    #具体的缺陷类别
    for i in range(split_gap):
        logger.info('S3D_Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))
    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mBFIoU, mAcc, allAcc))
    #前景与后景
    for i in range(args.classes):
        logger.info('BF_Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mBFIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
    print(1)









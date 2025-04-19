import random

import numpy as np
import math
import numbers
import collections
import cv2
import jittor as jt

manual_seed = 2022
random.seed(manual_seed)
jt.set_global_seed(manual_seed)

class Compose(object):
    # 组合多个变换操作
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label):
        for t in self.segtransform:
            image, label = t(image,label)
        return image, label

class ToVar(object):
    def __call__(self, image, label):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        # H * W * c => c * H * W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image = jt.array(image)
        label = jt.array(label.astype(np.int64))
        return image, label

class Normalize(object):
    # 对图像的每个通道进行标准化操作：(channel - mean) / std
    def __init__(self, mean, std = None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = jt.reshape(jt.array(mean),(-1,1,1)) # reshape 成 (C,1,1)
        self.std = jt.reshape(jt.array(std),(-1,1,1)) if std is not None else None

    def __call__(self, image, label):
        if self.std is None:
            image = image - self.mean
        else:
            image = (image - self.mean) / self.std
        return image, label

class Resize(object):
    # 将输入图像和标签调整到给定的尺寸
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        def find_new_hw(ori_h, ori_w, test_size):
            # 取原始长宽 较大者进行调整
            if ori_h >= ori_w:
                ratio = test_size * 1.0 / ori_h
                new_h = test_size
                new_w = int(ori_w * ratio)
            elif ori_w > ori_h:
                ratio = test_size * 1.0 / ori_w
                new_h = int(ori_h * ratio)
                new_w = test_size

            if new_h % 8 != 0:
                new_h = (int(new_h / 8)) * 8
            else:
                new_h = new_h
            if new_w % 8 != 0:
                new_w = (int(new_w / 8)) * 8
            else:
                new_w = new_w
            return new_h, new_w

        # 调整图像尺寸
        test_size = self.size
        new_h , new_w = find_new_hw(image.shape[0],image.shape[1],test_size)
        image_crop = cv2.resize(image, dsize=(int(new_w),int(new_h)),interpolation=cv2.INTER_LINEAR)
        #填充
        back_crop = np.zeros((test_size, test_size, 3))
        back_crop[:new_h,:new_w,:] = image_crop
        image = back_crop

        # 调整分割掩码的尺寸
        s_mask = label
        new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
        s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask = np.ones((test_size, test_size)) * 255
        back_crop_s_mask[:new_h, :new_w] = s_mask
        label = back_crop_s_mask

        return image,label

class test_Resize(object):
    # 将输入图像和标签调整为给定的大小
    #和Resize的区别是更加严格控制大小，如果原尺寸大于缩放的尺寸则不进行
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        def find_new_hw(ori_h, ori_w, test_size):
            if max(ori_h, ori_w) > test_size:
                if ori_h >= ori_w:
                    ratio = test_size * 1.0 / ori_h
                    new_h = test_size
                    new_w = int(ori_w * ratio)
                elif ori_w > ori_h:
                    ratio = test_size * 1.0 / ori_w
                    new_h = int(ori_h * ratio)
                    new_w = test_size

                if new_h % 8 != 0:
                    new_h = (int(new_h / 8)) * 8
                else:
                    new_h = new_h
                if new_w % 8 != 0:
                    new_w = (int(new_w / 8)) * 8
                else:
                    new_w = new_w
                return new_h, new_w
            else:
                return ori_h, ori_w

        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)
        if new_w != image.shape[0] or new_h != image.shape[1]:
            image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        else:
            image_crop = image.copy()
        back_crop = np.zeros((test_size, test_size, 3))
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop

        s_mask = label
        new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
        if new_w != s_mask.shape[0] or new_h != s_mask.shape[1]:
            s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),
                                interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask = np.ones((test_size, test_size)) * 255
        back_crop_s_mask[:new_h, :new_w] = s_mask
        label = back_crop_s_mask

        return image,label

class RandScale(object):
    # 随机缩放图像与标签
    def __init__(self,scale, aspect_ratio=None):
        #scale aspect_ratio 是一个元组（min,max）
        self.scale = scale
        self.aspect_ratio = aspect_ratio


    def __call__(self, image, label):
        #随机选取一个缩放因子
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_rate = 1.0

        if self.aspect_ratio is not None:
            temp_aspect_rate = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_rate = math.sqrt(temp_aspect_rate)

        #公式推导：宽高比限制 + 面积比限制
        scale_factor_x = temp_scale * temp_aspect_rate
        scale_factor_y = temp_scale / temp_aspect_rate
        image = cv2.resize(image, None, fx=scale_factor_x,fy=scale_factor_y,interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label

class Crop(object):
    #裁剪给定的图像和标签
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        self.size = size
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))

        # padding 必须是 3 list
        self.padding = padding
        self.ignore_label = ignore_label

    def __call__(self, image, label):
        h, w = label.shape

        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        #padding
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            # 对图像进行边缘填充
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        raw_label = label
        raw_image = image

        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        # 裁剪图像和标签
        image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
        # 统计正样本
        raw_pos_num = np.sum(raw_label == 1)
        pos_num = np.sum(label == 1)
        crop_cnt = 0
        while (pos_num < 0.85 * raw_pos_num and crop_cnt <= 30):
            image = raw_image
            label = raw_label
            if self.crop_type == 'rand':
                h_off = random.randint(0, h - self.crop_h)
                w_off = random.randint(0, w - self.crop_w)
            else:
                h_off = int((h - self.crop_h) / 2)
                w_off = int((w - self.crop_w) / 2)
            image = image[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            label = label[h_off:h_off + self.crop_h, w_off:w_off + self.crop_w]
            raw_pos_num = np.sum(raw_label == 1)
            pos_num = np.sum(label == 1)
            crop_cnt += 1

        #resize 到目标大小
        if image.shape != (self.size[0], self.size[0], 3):
            image = cv2.resize(image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)

        return image, label

class RandRotate(object):
    # 随机旋转图像和标签，旋转角度范围在 [rotate_min, rotate_max] 之间
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):

        self.rotate = rotate
        self.padding = padding
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            # 计算旋转矩阵
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.ignore_label)
        return image, label

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p # 翻转概率

    def __call__(self, image, label):
        # 以 p 的概率进行水平翻转
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label
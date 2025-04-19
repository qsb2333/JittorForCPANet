# CPANet-Jittor (Jittor Implementation of [CPANet](https://ieeexplore.ieee.org/document/10049179))

[![Jittor](https://img.shields.io/badge/Jittor-1.3.9.14-blue)](https://cg.cs.tsinghua.edu.cn/jittor/)

## 目录
- [环境配置](#环境配置)
- [数据集FSSD-12](#数据集FSSD-12)
- [训练与测试脚本](#训练与测试脚本)
- [基准实验](#基准实验)
  - [Backbone对比实验](#backbone对比实验)
  -  [loss曲线](#loss曲线)
- [消融实验](#消融实验)
- [复现过程中Jittor与PyTorch关键区别](#复现过程中jittor与pytorch关键区别)

## 环境配置

```yaml
GPU:        NVIDIA RTX 4090D (24GB)
CPU:        Intel Xeon Platinum 8474C (15 vCPU)
Platform:   AutoDL云平台
Python == 3.8.20
CUDA == 11.3
Jittor == 1.3.9.14
```

---

## 数据集
### 数据划分
| Fold | 缺陷类别 |
|------|----------|
| 0    | abrasion-mask, iron-sheet-ash, liquid, oxide-scale |
| 1    | oil-spot, water-spot, patch, punching |
| 2    | red-iron-sheet, scratch, roll-printing, inclusion |

### 数据下载
- 百度网盘: [链接](https://pan.baidu.com/s/1dEai3yXrFOsuWcQ5mkE7_A?pwd=qzo6) 提取码: `qzo6`
```text
  ├──CPANet/
  └──FSSD-12/
	  ├── Steel_Am/
	  |   	├── GT/
	  |   	├── Images/
	  |   	└── Nd/
	  ├── Steel_la/
	  ├── Steel_Ld/
	  ├── Steel_Op/
	  ├── Steel_Os/
	  ├── Steel_Pa/
	  ├── Steel_Pk/
	  ├── Steel_Ri/
	  ├── Steel_Rp/
	  ├── Steel_Sc/
	  ├── Steel_Se/
	  └── Steel_Ws/
```

---

## 训练与测试脚本
```bash
python train.py --config config/SSD/fold0_vgg16.yaml
python train.py --config config/SSD/fold1_resnet50.yaml
```

---

## 基准实验
### 📌 注意事项

- 原论文未提供其 ResNet 变体的预训练模型参数。因此，表格中使用的 **ResNet-50** 实验结果是基于 **标准 ImageNet 预训练参数** 的，可能与原论文报告的结果存在较大差异，仅供参考。
  
- 此外，实验过程中发现该模型在测试时存在一定程度的**结果波动**，并非完全稳定。例如，在 **VGG16-Fold0** 实验中，`mIoU` 值会在 **0.4978 ~ 0.5214** 范围之间变化（此范围为多次测试所得上下界）。


### Backbone对比实验
| Backbone | Method | MIoU (1-shot) |               |               | Mean  | FB-IoU (1-shot) | MIoU (5-shot) |               |               | Mean  | FB-IoU (5-shot) |
|----------|--------|---------------|---------------|---------------|-------|-----------------|---------------|---------------|---------------|-------|-----------------|
|          |        | Fold0        | Fold1        | Fold2        |       |                   | Fold0        | Fold1        | Fold2        |       |                 |
| VGG16    | Ours   | 52.14          | 65.8          | 63.4          | 65.5  | 73.2            | 72.1          | 70.5          | 69.8          | 70.8  | 78.4            |
| ResNet50 | Ours   | 68.9          | 67.2          | 65.1          | 67.1  | 74.6            | 73.8          | 72.1          | 71.3          | 72.4  | 79.9            |

---
### loss曲线
本部分展示了以 **VGG16** 作为主干网络（Backbone）在 **Fold 0** 条件下训练和测试过程中记录的 **Loss 曲线**，用于直观观察模型的收敛情况和性能变化。

- **训练集 Loss 曲线：**  
  ![Train Loss Curve](./results/loss_curves/vgg16_fold0_train.png)

- **验证集 Loss 曲线：**  
  ![Validation Loss Curve](./results/loss_curves/vgg16_fold0_val.png)

> 💡 曲线图文件默认保存在：`results/loss_curves/` 目录下。
---
## 消融实验
## 🔬 消融实验：模块间组件分析（基于 VGG16）

本部分展示了在以 **VGG16** 为主干网络的前提下，对模型中的关键模块（CPP、SA、SSA）进行组件级别的消融实验。

### 🧪 实验设定

- **Baseline 设置：**  
  - 未使用 CPP 模块，而是采用**掩码全局平均池化**（Masked GAP）提取缺陷前景信息；
  - 移除了 SA 模块；
  - 解码器由**全卷积解码器**替代了原有的**空间注意力挤压解码器**。

- **实验方式：**  
  仅展示 **1-shot** 下的模型性能，指标包括 `Mean IoU (MIoU)` 和 `Foreground-Background IoU (FB-IoU)`。

### 模块消融实验结果（1-Shot）

| CPP | SA  | SSA | MIoU-Fold0 | MIoU-Fold1 | MIoU-Fold2 | MIoU-Mean | FB-IoU |
|:----:|:----:|:----:|:------------:|:------------:|:------------:|:-----------:|:--------:|
| ✗   | ✗   | ✗   | 57.7        | 55.1        | 48.9        | 53.9       | 71.4    |
| ✓    | ✗   | ✗   | 58.3        | 60.9        | 49.9        | 56.4       | 72.6    |
| ✓   | ✓    | ✗   | 65.8        | 63.3        | 54.4        | 61.2       | 75.9    |
| ✗   | ✗  | ✓    | 59.3        | 58.6        | 50.6        | 56.2       | 71.9    |
| ✓   | ✓   | ✓   | 66.0        | 64.0        | 54.6        | **61.5**   | **76.1** |

>  **说明：**
> - `✓` 表示该模块被启用，`✗` 表示未启用；



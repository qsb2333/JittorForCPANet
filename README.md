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
- [复现过程中Jittor框架的错误和与PyTorch关键区别](#复现过程中Jittor框架的错误和与pytorch关键区别)
- [总结](#总结)

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
### 注意事项

- 原论文未提供其 ResNet 变体的预训练模型参数。因此，表格中使用的 **ResNet-50** 实验结果是基于 **标准 ImageNet 预训练参数** 的，可能与原论文报告的结果存在较大差异，仅供参考。
  
- 此外，实验过程中发现该模型在测试时存在一定程度的**结果波动**，并非完全稳定。例如，在 **VGG16-Fold0** 实验中，`mIoU` 值会在 **0.4978 ~ 0.5214** 范围之间变化（此范围为多次测试所得上下界），因此可能与原论文报告的结果存在差异，仅供参考。


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
  ![Train Loss Curve](./loss_curves/loss_train.png)

- **验证集 Loss 曲线：**  
  ![Validation Loss Curve](./loss_curves/val_train.png)

---
## 消融实验
## 模块间组件分析（基于 VGG16）

本部分展示了在以 **VGG16** 为主干网络的前提下，对模型中的关键模块（CPP、SA、SSA）进行组件级别的消融实验。

### 实验设定

- **Baseline 设置：**  
  - 未使用 CPP 模块，而是采用**掩码全局平均池化**（Masked GAP）提取缺陷前景信息；
  - 移除了 SA 模块；
  - 解码器由**全卷积解码器**替代了原有的**空间注意力挤压解码器**。

- **实验方式：**  
  仅展示 **1-shot** 下的模型性能，指标包括 `Mean IoU (MIoU)` 和 `Foreground-Background IoU (FB-IoU)`。
 ---

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
---

## 辅助 Loss 超参数 k 的选择（基于 VGG16）

我们的总损失函数由两个独立的损失项线性组合而成，其中超参数 $k$ 控制辅助损失（SA分支）的权重。为确定最佳的超参数设置，我们在 $k \in \{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ 范围内进行了消融实验。

> 当 $k=0$ 时，SA 分支将完全失效，相当于移除辅助损失路径。

### 不同 k 值下的模型性能（1-Shot）

| k   | MIoU-Fold0 | MIoU-Fold1 | MIoU-Fold2 | MIoU-Mean | FB-IoU |
|:----:|:------------:|:------------:|:------------:|:------------:|:--------:|
| 0.0 | 61.8        | 60.3        | 50.4        | 57.5       | 72.9    |
| 0.2 | 65.2        | 61.0        | 53.0        | 59.7       | 74.8    |
| 0.4 | **66.0**    | **64.0**    | **54.6**    | **61.5**   | **76.1** |
| 0.6 | 63.4        | 62.9        | 52.9        | 59.7       | 74.8    |
| 0.8 | 62.8        | 61.4        | 55.2        | 59.8       | 74.5    |
| 1.0 | 63.8        | 62.4        | 50.6        | 58.9       | 74.7    |
---
## 复现过程中Jittor框架的错误和与PyTorch关键区别


在使用 Jittor 框架复现 CPANet 的过程中，发现了一些与 PyTorch 不一致的实现细节与潜在错误。以下内容将逐一介绍这些差异及其对结果的影响，并配以对比代码说明。

---

### 1、`jittor.misc.histc` 源码存在边界处理问题

Jittor 中 `histc` 函数在处理边界值（例如最大值等于 `max` 的情况）时，行为与 PyTorch 不一致。

#### 问题描述：

- **输入数据**：`[0, 0, 1, 1, 0, 1, 2, 2, 2]`
- **参数设置**：`bins=3`, `min=0`, `max=2`

#### PyTorch 输出：

```python
import torch
data = torch.tensor([0, 0, 1, 1, 0, 1, 2, 2, 2], dtype=torch.float32)
hist = torch.histc(data, bins=3, min=0, max=2)
print(hist)  # tensor([3., 3., 3.])
````

#### Jittor 输出：

```python
import jittor as jt
data = jt.array([0, 0, 1, 1, 0, 1, 2, 2, 2])
hist = jt.misc.histc(data, bins=3, min=0, max=2)
print(hist)  # [3. 3. 1.]
```

 **原因**：Jittor 的实现没有将 `max` 本身计入统计中，导致最后一桶数据缺失。

----------

### 2、`max` 与 `argmax` 返回值结构不同

Jittor 与 PyTorch 在 `max` 和 `argmax` 函数的返回格式上存在明显区别。

####  PyTorch：

```python
import torch

x = torch.tensor([[1.0, 5.0, 3.0]])
value, index = torch.max(x, dim=1)
print(value)  # tensor([5.])
print(index)  # tensor([1])

argmax_index = torch.argmax(x, dim=1)
print(argmax_index)  # tensor([1])
```

#### Jittor：

```python
import jittor as jt

x = jt.array([[1.0, 5.0, 3.0]])
value = jt.max(x, dim=1)
print(value)  # [5.] ← 仅返回值

index, val = jt.argmax(x, dim=1)
print(index)  # [1]
print(val)    # [5.] ← 注意：argmax 返回下标和对应值
```

 **区别总结**：
| 功能        | PyTorch 返回内容        | Jittor 返回内容        |
|-------------|--------------------------|-------------------------|
| `max`       | `(value, index)`         | `value`                |
| `argmax`    | `index`                  | `(index, value)`       |

----------

### 3、Loss 反向传播方式不同

Jittor 中使用 `optimizer.step(loss)`，而不是像 PyTorch 中的 `loss.backward()` 和 `optimizer.step()` 两步式。

#### PyTorch：

```python
loss = criterion(output, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

#### Jittor：

```python
loss = criterion(output, target)
optimizer.step(loss)  # 自动完成 backward + update
```
----------
## 总结

本次实验基于 **Jittor 框架** 复现了 **[CPANet](https://ieeexplore.ieee.org/document/10049179)** 网络结构。实验过程中，虽然成功实现了 CPANet 的整体网络逻辑和训练流程，但复现效果仍存在一定的差距，可能由以下因素造成：

### 存在问题与挑战

- 对 Jittor 框架的使用尚不够熟练，尤其在某些函数接口与 PyTorch 的差异方面（详见前述差异分析部分）；
- 缺乏原论文中特定 **预训练参数**，本实验采用标准 ResNet50 权重作为替代，可能影响最终性能；
- 个人能力与调试经验尚显不足，在参数调优、调试策略等方面仍有待提升；
- 模型测试结果存在一定波动。例如，VGG16 作为 backbone 在 fold-0 上的 mIoU 浮动范围为 `0.4978 ~ 0.5214`，其余 folds 也表现出类似的波动情况。这种不稳定性可能与模型结构、训练策略或框架实现细节有关。

### 收获与成长

- **深入理解了 CPANet 的模块设计思想与注意力机制实现**，对 Few-Shot 分割任务的关键技术有了更深刻认识；
- **掌握了 Jittor 框架的基本用法与模型构建流程**，包括数据加载、模型定义、训练与测试过程；
- 积累了进行复现类研究项目的实践经验，对未来开展深度学习模型设计与复现具有重要指导意义；
- 熟悉了从 PyTorch 到 Jittor 的迁移过程，增强了框架迁移与兼容调试能力。

---


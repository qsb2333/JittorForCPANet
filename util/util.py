import jittor as jt
import yaml


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n = 1):
        self.val = val
        self.sum +=val * n
        self.count +=n
        self.avg = self.sum / self.count

def load_cfg_from_yamlFile(file):
    cfg={}
    with open(file,'r') as f:
        cfg_from_yamlfile = yaml.safe_load(f)

    for key in cfg_from_yamlfile:
        for k, v in cfg_from_yamlfile[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=4, scale_lr=10., warmup=False, warmup_step=500):
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    if curr_iter % 50 == 0:
        print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr


def histc(input, bins, min=0., max=0.):
    '''
    计算输入的N维数组的直方图。
    '''
    if min == 0 and max == 0:
        min, max = input.min(), input.max()
    assert min < max
    bin_length = (max - min) / bins
    histc = jt.floor((input[jt.logical_and(input >= min, input <= max)] - min) / bin_length)
    histc = jt.minimum(histc, bins - 1).int().reshape(-1)  # 修正关键行
    #["@e0(i0)"] 映射规则 用 histc[i0] 作为输出下标
    hist = jt.ones_like(histc).float().reindex_reduce("add", [bins, ], ["@e0(i0)"], extras=[histc])
    if hist.sum() != histc.shape[0]:
        hist[-1] += 1
    return hist

def intersectionAndUnion(output, target, K, ignore_index=255):
    #计算交集、并集和target区域
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = histc(intersection,bins=K , min=0, max = K- 1)
    area_output = histc(output, bins=K, min=0, max=K - 1)
    area_target = histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection #A并B = A + B - A交B
    return area_intersection , area_union, area_target


class CfgNode(dict):
    def __init__(self, init_dict=None,key_list=None):
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self,name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name,value):
        self[name] =value

    def __str__(self):
        #美化输出
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ")+line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k),seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)

        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())
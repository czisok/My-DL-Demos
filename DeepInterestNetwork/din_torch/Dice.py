"""
Dice Activation (Data Adaptive Individualized Activation) - PyTorch 版
==========================================================================
与 TF1 版 din/Dice.py 保持完全一致的建模逻辑：
  Dice(x) = alpha * (1 - p) * x + p * x
  其中 p = sigmoid(beta * BN(x))   （BN 不带 center/scale）

TF1 → PyTorch 对应：
  tf.get_variable('alpha', shape=[C])        → nn.Parameter(torch.zeros(C))
  tf.get_variable('beta',  shape=[C])        → nn.Parameter(torch.zeros(C))
  tf.layers.batch_normalization(_x, center=False, scale=False)
                                             → nn.BatchNorm1d(C, affine=False)
  tf.variable_scope(..., reuse=AUTO_REUSE)   → 同一个 DiceLayer 对象多次 forward 自动复用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLayer(nn.Module):
    """
    Dice 激活层。逻辑与 TF1 版完全相同：
      1. 对输入沿最后一维做 BN（不带 gamma/beta/affine）
      2. x_p = sigmoid(beta * x_normed)
      3. out = alpha * (1 - x_p) * x + x_p * x
    """

    def __init__(self, num_features, epsilon=1e-9, name=''):
        super().__init__()
        self._name = name
        self.epsilon = epsilon
        # TF1 版 BN: center=False, scale=False → affine=False (PyTorch)
        # 注意：TF 的 BatchNormalization 默认 axis=-1（层归一化的 axis），
        #       对应 PyTorch 中必须把最后一维移到通道维做 BN1d，
        #       即先把 [..., C] permute 成 [N, C, ...]，这里用的方法是：
        #       把 shape 变形成 [-1, C] 做 BN1d，再 reshape 回去。
        self.bn = nn.BatchNorm1d(num_features, affine=False, eps=epsilon)
        # alpha 和 beta 初始化为 0（对应 TF1 的 constant_initializer(0.0)）
        self.alphas = nn.Parameter(torch.zeros(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 可能是 [B, C] 或 [B, T, C] 或 [B, N, T, C]
        # 统一把最后一维当作 C，其余 flatten 到 batch 维
        ori_shape = x.shape
        C = ori_shape[-1]
        x_flat = x.reshape(-1, C)                       # [*, C]
        x_normed = self.bn(x_flat)                       # [*, C]
        x_normed = x_normed.reshape(ori_shape)           # 还原成原 shape
        x_p = torch.sigmoid(self.beta * x_normed)        # 逐元素
        return self.alphas * (1.0 - x_p) * x + x_p * x


class PReLULayer(nn.Module):
    """
    对应原 TF1 parametric_relu()：pos = relu(x);  neg = alpha * (x - |x|) * 0.5
    """

    def __init__(self, num_features, name=''):
        super().__init__()
        self._name = name
        self.alphas = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = F.relu(x)
        neg = self.alphas * (x - x.abs()) * 0.5
        return pos + neg


# 为兼容原 TF1 版 dice(_x, name='xxx') 函数式调用，保留相同入口
_DICE_CACHE = {}


def dice(_x: torch.Tensor, axis=-1, epsilon=1e-9, name='') -> torch.Tensor:
    """
    函数式 Dice 入口。按 name 复用同一个 DiceLayer（对应原 TF1 的 reuse=AUTO_REUSE）。
    注意：必须把返回的 DiceLayer 注册到调用方 model 的子模块上，否则参数不会被优化器收录。
    因此建议直接实例化 DiceLayer。
    """
    C = _x.shape[axis]
    key = (name, C)
    if key not in _DICE_CACHE:
        _DICE_CACHE[key] = DiceLayer(C, epsilon=epsilon, name=name)
    return _DICE_CACHE[key](_x)


def parametric_relu(_x: torch.Tensor, name='') -> torch.Tensor:
    C = _x.shape[-1]
    key = ('prelu', name, C)
    if key not in _DICE_CACHE:
        _DICE_CACHE[key] = PReLULayer(C, name=name)
    return _DICE_CACHE[key](_x)

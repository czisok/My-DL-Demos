from __future__ import annotations
import math
from typing import List

import torch
import torch.nn.functional as F


def sequence_mask(
    lengths: torch.Tensor,
    maxlen: int | None = None,
    dtype: torch.dtype = torch.bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    等价于 TensorFlow 里的 tf.sequence_mask。

    Args:
        lengths: [B]  int64  每条样本的有效长度（>= 0）
        maxlen:  int/None  如果为 None 则取 lengths.max()；否则按给定长度构造 mask
        dtype:   输出 dtype，默认 bool （True=有效位置，False=padding 位置）
                 也可以传 torch.float32 得到 (1.0/0.0) 的数值 mask，可直接乘到张量上
        device:  输出 device；传 None 时使用 lengths 的 device

    Returns:
        mask:  [B, maxlen]
            mask[b, t] = True (或 1.0) 当且仅当 t < lengths[b]
            mask[b, t] = False(或 0.0) 当 t >= lengths[b]

    Examples:
        >>> sl = torch.tensor([2, 3, 1], dtype=torch.long)
        >>> sequence_mask(sl, maxlen=5)
        tensor([[ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [ True, False, False, False, False]])
    """
    if device is None:
        device = lengths.device
    if maxlen is None:
        maxlen = int(lengths.max().cpu().item()) if lengths.numel() > 0 else 0

    # arange([0,1,2,...,maxlen-1]) 扩成 [1, maxlen]，lengths 扩成 [B,1] → 逐元素比较
    idx = torch.arange(maxlen, device=device, dtype=lengths.dtype).unsqueeze(0)   # [1, M]
    l = lengths.to(device=device).unsqueeze(1)                                     # [B, 1]
    mask_bool = idx < l                                                             # [B, M]

    if dtype == torch.bool:
        return mask_bool
    return mask_bool.to(dtype)


def clip_by_global_norm_tf_(params, clip_norm: float, norm_type: float = 2.0) -> float:
    """TF 等价的 clip_by_global_norm：同时支持 dense 与 sparse_coo 梯度，对齐 TF IndexedSlices 的聚合方式。

    - dense 梯度：对整个张量展平计算 L2 范数（与 TF dense 相同）。
    - sparse 梯度（sparse_coo）：仅对 coalesce 后的 values() 计算 L2 范数，与 TF IndexedSlices
      只对 indices 对应的 dense 切片聚合 global_norm 的行为严格一致。
    - 缩放规则：scale = clip_norm / max(global_norm, clip_norm)；global_norm <= clip_norm 时不做任何修改。

    Args:
        params: Iterable[torch.nn.Parameter]，通常传 model.parameters()
        clip_norm: float，裁剪阈值（与 tf.clip_by_global_norm(t_list, clip_norm) 一致）
        norm_type: float，范数阶数，默认 2（L2 global norm）

    Returns:
        global_norm: float，裁剪前的总体范数（用于日志打印 debug）
    """
    assert norm_type == 2.0, f'当前仅实现了 L2 global norm，得到 norm_type={norm_type}'
    if clip_norm <= 0:
        return 0.0

    total_sq = 0.0
    param_grad_pairs: List = []

    # ---------- Phase 1：聚合 global_norm ----------
    for p in params:
        g = p.grad
        if g is None:
            continue
        if g.layout == torch.sparse_coo:
            # 对齐 TF IndexedSlices：只对 values()（被 lookup 到的行对应的梯度切片）参与范数
            gc = g.coalesce()
            v = gc.values().detach()
            norm_v = torch.linalg.vector_norm(v, ord=2).item()
            total_sq += norm_v * norm_v
            param_grad_pairs.append((p, 'sparse', None))
        else:
            gd = g.detach()
            norm_d = torch.linalg.vector_norm(gd.reshape(-1), ord=2).item()
            total_sq += norm_d * norm_d
            param_grad_pairs.append((p, 'dense', None))

    global_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
    if global_norm <= clip_norm:
        return float(global_norm)

    scale = clip_norm / global_norm

    # ---------- Phase 2：按比例原地缩放 .grad ----------
    # 注意：sparse param 的 grad 不要直接 inplace mul，先 detach values 再写回避免 autograd 问题
    with torch.no_grad():
        for p, kind, _ in param_grad_pairs:
            g = p.grad
            if g is None:
                continue
            if kind == 'sparse':
                gc = g.coalesce()
                new_values = gc.values().mul_(scale)
                new_grad = torch.sparse_coo_tensor(
                    indices=gc.indices(),
                    values=new_values,
                    size=gc.size(),
                    dtype=gc.dtype,
                    device=gc.device,
                ).coalesce()
                p.grad = new_grad
            else:
                g.mul_(scale)

    return float(global_norm)

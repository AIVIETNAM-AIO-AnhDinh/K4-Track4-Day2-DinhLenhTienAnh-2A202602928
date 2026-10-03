"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOSS_CHOICES = ("ce", "ls", "focal", "ce_weighted")


def build_criterion(kind: str = "ce", smoothing: float = 0.1, gamma: float = 2.0,
                    alpha=None, weight=None):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    "ce_weighted" cần `weight` (tensor 9 phần tử, xem class_weights). Mọi loss lấy trung bình batch.
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing)
    if kind == "focal":
        return FocalLoss(gamma, alpha)
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted cần weight (losses.class_weights)")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"loss={kind!r} không hợp lệ, chọn một trong {LOSS_CHOICES}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt: loss = (1 - eps) * (-log p_y) + eps * mean_k(-log p_k). eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=1)
        return ((1 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp. gamma = 0 và alpha = None cho đúng cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float))

    def forward(self, logits, target):
        logp_t = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: tỉ lệ nghịch với số ảnh (1 / n_c)
    - beta > 0: class-balanced, w_c = (1 - beta) / (1 - beta ** n_c) (Cui et al. arXiv:1901.05555)
    Cả hai được chuẩn hoá để tổng trọng số bằng số lớp (trung bình 1).
    """
    n = torch.as_tensor(np.asarray(counts), dtype=torch.float64)
    if (n <= 0).any():
        raise ValueError("Mọi lớp phải có ít nhất 1 ảnh train")
    w = 1.0 / n if beta == 0 else (1 - beta) / (1 - beta ** n)
    return (w / w.sum() * len(n)).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn; lam ~ Beta(alpha, alpha), dùng RNG của numpy (đã seed).

    - "mixup":  x_mix = lam * x + (1 - lam) * x[perm]
    - "cutmix": dán một hộp từ x[perm] vào x; lam được tính lại theo DIỆN TÍCH THỰC của hộp
                sau khi bị cắt ở biên (slide trang 48)
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        cut = np.sqrt(1.0 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip([cy - ch // 2, cy + ch // 2], 0, h)
        x1, x2 = np.clip([cx - cw // 2, cx + cw // 2], 0, w)
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode={mode!r} không hợp lệ, chọn 'mixup' hoặc 'cutmix'")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

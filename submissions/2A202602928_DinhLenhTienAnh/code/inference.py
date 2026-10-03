"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views)      -> (filenames, y_true, {tên view: logits})
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def _forward(model, x, device, amp: bool, channels_last: bool):
    x = x.to(device, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        return model(x).float().cpu()


def predict_logits(model, loader, device, view=None, amp: bool = True, channels_last: bool = True):
    """Chạy model trên loader, gom logit theo đúng thứ tự file. `view`: hàm biến đổi batch hoặc None."""
    names, y, out = predict_views(model, loader, device, {"v": view or view_identity}, amp, channels_last)
    return names, y, out["v"]


def predict_views(model, loader, device, views: dict, amp: bool = True, channels_last: bool = True):
    """Một lượt qua loader, mỗi batch chạy mọi view trong `views` ({tên: hàm(batch) -> batch}).

    Trả về (filenames, y_true, {tên: logits[N, 9]}). Một view lỗi (ví dụ ViT ở độ phân giải khác)
    thì logits của nó là None và lỗi được ghi trong out["_errors"].
    """
    model.eval()
    names, ys = [], []
    outs = {k: [] for k in views}
    errors = {}
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)   # view được tạo trên GPU
            for k, fn in views.items():
                if k in errors:
                    continue
                try:
                    outs[k].append(_forward(model, fn(x), device, amp, channels_last))
                except Exception as e:  # noqa: BLE001 - ghi lại để báo "không áp dụng được"
                    errors[k] = f"{type(e).__name__}: {e}"[:300]
            ys.append(y)
            names.extend(f)
    res = {k: (None if k in errors else torch.cat(v).numpy()) for k, v in outs.items()}
    res["_errors"] = errors
    return names, torch.cat(ys).numpy(), res


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop`; flip=True thêm bản lật của từng crop. Trả về list batch."""
    h, w = x.shape[-2:]
    tops = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), ((h - crop) // 2, (w - crop) // 2)]
    crops = [x[..., t:t + crop, l:l + crop] for t, l in tops]
    return crops + [view_hflip(c) for c in crops] if flip else crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias khi thu nhỏ). Trả về list batch.

    CNN có global pooling nhận được mọi kích thước. ViT/DeiT dựng với img_size cố định và Swin
    (cửa sổ 7x7) thì báo lỗi ở kích thước khác 224: predict_views ghi lại là không áp dụng được.
    """
    return [x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear", antialias=True,
                                                     align_corners=False) for s in sizes]


def softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view của TTA thành xác suất (N, 9).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    stack = np.stack([np.asarray(z, dtype=np.float64) for z in logits_per_view])
    if space == "prob":
        return softmax(stack).mean(0)
    if space == "logit":
        return softmax(stack.mean(0))
    raise ValueError(f"space={space!r} không hợp lệ, chọn 'prob' hoặc 'logit'")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình trên CÙNG tập ảnh, cùng thứ tự file."""
    shapes = {np.shape(p) for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"Các mô hình có số ảnh khác nhau: {shapes}")
    return np.mean([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL của softmax(logit / T) (slide trang 69). LBFGS trên log T.

    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return softmax(np.asarray(logits, dtype=np.float64) / T)


def _fused_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(device=conv.weight.device, dtype=conv.weight.dtype)
    gamma = bn.weight if bn.weight is not None else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.bias is not None else torch.zeros_like(bn.running_mean)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    scale = gamma / torch.sqrt(bn.running_var + bn.eps)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
        fused.bias.copy_(beta + (bias - bn.running_mean) * scale)
    return fused


def _set_submodule(root: nn.Module, target: str, module: nn.Module) -> None:
    parent, _, name = target.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, name, module)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Dò đồ thị bằng torch.fx để bắt mọi cặp Conv2d -> BatchNorm2d (kể cả cặp thuộc tính như
    conv1/bn1 của ResNet). BatchNormAct2d của timm (BN + activation) được thay bằng activation
    của nó sau khi gộp phần BN. Trả về GraphModule mới (model gốc không đổi), thuộc tính
    `n_fused` = số cặp đã gộp. Model không có BN (ViT, Swin, ConvNeXt) trả về n_fused = 0.
    Kiểm tra: so đầu ra trước/sau (lệch cỡ 1e-5 trở xuống ở FP32).
    """
    import torch.fx as fx

    class _Tracer(fx.Tracer):  # BatchNormAct2d nằm ngoài torch.nn: giữ nó là một module lá
        def is_leaf_module(self, m, qualname):
            return isinstance(m, nn.BatchNorm2d) or super().is_leaf_module(m, qualname)

    model = copy.deepcopy(model).eval()
    if not any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
        model.n_fused = 0
        return model
    gm = fx.GraphModule(model, _Tracer().trace(model))
    modules = dict(gm.named_modules())
    n = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_module" or not isinstance(modules.get(node.target), nn.BatchNorm2d):
            continue
        prev = node.args[0]
        if not (isinstance(prev, fx.Node) and prev.op == "call_module"
                and type(modules.get(prev.target)) is nn.Conv2d and len(prev.users) == 1):
            continue
        bn = modules[node.target]
        _set_submodule(gm, prev.target, _fused_conv(modules[prev.target], bn))
        if type(bn) is nn.BatchNorm2d:
            node.replace_all_uses_with(prev)
            gm.graph.erase_node(node)
        else:  # timm BatchNormAct2d: BN -> drop -> act; giữ drop + act
            _set_submodule(gm, node.target, nn.Sequential(bn.drop, bn.act))
        n += 1
    gm.graph.lint()
    gm.recompile()
    gm.delete_all_unused_submodules()
    gm.n_fused = n
    return gm

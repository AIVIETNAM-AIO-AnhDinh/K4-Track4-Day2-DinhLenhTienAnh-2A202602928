"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init, head_std) -> nn.Module
    reset_head(model, std)                                        -> None
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (giữ BN ở eval nếu đóng băng)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
    weight_tag(model) -> str                 data_config(model) -> dict (mean/std của trọng số)
"""
from __future__ import annotations

import timm
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}
INIT_CHOICES = ("scratch", "frozen", "finetune")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", head_std: float | None = 1e-3):
    """Tạo model phân loại 9 lớp qua timm (head mới khởi tạo ngẫu nhiên).

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng backbone, chỉ train head
      - "finetune" : pretrained=True, train toàn bộ
    `head_std`: khởi tạo lại head bằng reset_head(model, head_std) cho MỌI backbone, để loss ban
    đầu ≈ ln 9 (slide trang 59). Init mặc định của timm cho EfficientNet/MobileNetV3 cho logit
    std ≈ 3 và loss ban đầu ≈ 5,6-5,9; std 0,01 vẫn lệch tới ~0,1 với Swin/ConvNeXt (đặc trưng
    sau LayerNorm có norm 20-40), nên dùng 1e-3 (cỡ head init của MAE khi fine-tune). None = giữ init của timm.
    Tag trọng số thực sự được tải: weight_tag(model).
    """
    if init not in INIT_CHOICES:
        raise ValueError(f"init={init!r} không hợp lệ, chọn một trong {INIT_CHOICES}")
    pretrained = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                              drop_rate=drop_rate)
    model.is_pretrained = pretrained
    model.frozen_backbone = False
    if head_std is not None:
        reset_head(model, head_std)
    if init == "frozen":
        freeze_backbone(model)
    return model


def reset_head(model, std: float = 1e-3) -> None:
    """Head: weight ~ N(0, std) cắt ở ±2 std, bias = 0 -> logit ban đầu ≈ 0, softmax ≈ đều."""
    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise TypeError(f"Head của {type(model).__name__} là {type(head).__name__}, không phải nn.Linear")
    nn.init.trunc_normal_(head.weight, std=std, a=-2 * std, b=2 * std)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


def weight_tag(model) -> str:
    """Tên đầy đủ của bộ trọng số, ví dụ 'resnet50.a1_in1k'; 'scratch' nếu không tải trọng số."""
    if not getattr(model, "is_pretrained", True):
        return "scratch"
    cfg = model.pretrained_cfg
    tag = cfg.get("tag")
    return f"{cfg['architecture']}.{tag}" if tag else cfg["architecture"]


def data_config(model) -> dict:
    """Cấu hình tiền xử lý của trọng số (mean, std, input_size, crop_pct, interpolation)."""
    return timm.data.resolve_data_config({}, model=model)


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    Backbone đóng băng thì BatchNorm cũng phải ở eval (không cập nhật running stats):
    trong train loop gọi set_train_mode(model) thay cho model.train().
    """
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone bị đóng băng thì đưa mọi BatchNorm về eval."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52.

    - backbone có ndim > 1: lr = lr_backbone, weight_decay = weight_decay
    - norm và bias của backbone (ndim <= 1), cùng pos_embed/cls_token (model.no_weight_decay()):
      lr = lr_backbone, weight_decay = 0
    - head mới: lr = lr_head, weight_decay = weight_decay
    Bỏ qua tham số requires_grad == False và nhóm rỗng.
    """
    head = _head_param_ids(model)
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    decay, no_decay, head_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head:
            head_params.append(p)
        elif p.ndim <= 1 or name in skip:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"name": "backbone", "params": decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"name": "backbone_no_decay", "params": no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"name": "head", "params": head_params, "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (có sẵn trong PyTorch), đếm FLOPs của
    conv/matmul/attention rồi chia 2 để ra MAC. Không đếm phép từng phần tử (BN, activation),
    nên có thể lệch vài phần trăm so với fvcore/ptflops.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 2 / 1e9

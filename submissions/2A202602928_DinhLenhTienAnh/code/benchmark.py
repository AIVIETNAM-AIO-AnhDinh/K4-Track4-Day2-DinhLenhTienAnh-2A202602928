"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (đọc ảnh, resize, chuẩn hoá): đầu vào là tensor đã nằm trên GPU
"""
from __future__ import annotations

import contextlib
import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây). `sync`: hàm đồng bộ (torch.cuda.synchronize) hoặc None trên CPU."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "n": iters}


def _prepare(model, dtype: str, device, channels_last: bool):
    m = copy.deepcopy(model).to(device).eval()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    if dtype == "fp16":
        m = m.half()
    ctx = (lambda: torch.autocast(device.type, dtype=torch.float16)) if dtype == "amp" else contextlib.nullcontext
    return m, ctx


def _timed_forward(m, ctx, make_input, warmup, iters, device):
    x = make_input()

    def fn():
        with torch.inference_mode(), ctx():
            m(x)

    return bench(fn, warmup, iters, torch.cuda.synchronize if device.type == "cuda" else None)


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, channels_last: bool = True, fused_bn: bool = False) -> dict:
    """Độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast FP16) | "fp16" (model.half()). Model gốc không bị thay đổi.
    Trả về một dòng cho sheet `Latency` của results.xlsx.
    """
    if dtype not in ("fp32", "amp", "fp16"):
        raise ValueError(f"dtype={dtype!r} không hợp lệ")
    device = torch.device(device)
    m, ctx = _prepare(model, dtype, device, channels_last)
    in_dtype = torch.float16 if dtype == "fp16" else torch.float32
    r = _timed_forward(m, ctx, lambda: torch.randn(batch_size, 3, img_size, img_size, device=device, dtype=in_dtype),
                       warmup, iters, device)
    del m
    return {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, "fused_bn": fused_bn,
            "channels_last": channels_last, **r, "images_per_s": batch_size / (r["p50"] / 1000),
            "torch": torch.__version__, "preprocessing": "không tính"}


def tta_latency(model, k_views: int, make_views, img_size: int = 256, dtype: str = "fp32",
                device: str = "cuda", warmup: int = 10, iters: int = 100, channels_last: bool = True) -> dict:
    """Độ trễ TTA K view cho MỘT ảnh: tạo K view trên GPU rồi chạy model trên batch K view.

    `make_views(x)` nhận (1, 3, img_size, img_size) và trả về tensor (K, 3, h, w). So sánh với
    K x p50 của một lượt (slide trang 63) ở notebook.
    """
    device = torch.device(device)
    m, ctx = _prepare(model, dtype, device, channels_last)
    in_dtype = torch.float16 if dtype == "fp16" else torch.float32
    x = torch.randn(1, 3, img_size, img_size, device=device, dtype=in_dtype)

    def fn():
        with torch.inference_mode(), ctx():
            v = make_views(x)
            if channels_last:
                v = v.contiguous(memory_format=torch.channels_last)
            m(v)

    r = bench(fn, warmup, iters, torch.cuda.synchronize if device.type == "cuda" else None)
    del m
    return {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": dtype, "batch": 1, "k_views": k_views, "img_size": img_size, **r,
            "images_per_s": 1 / (r["p50"] / 1000), "torch": torch.__version__, "preprocessing": "không tính"}

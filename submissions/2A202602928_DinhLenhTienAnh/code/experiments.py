"""experiments.py - điều phối thí nghiệm cho notebook: chạy nhiều cấu hình, nạp checkpoint, view TTA.

    run_many(cfgs, n_parallel)        -> list[summary]  (n_parallel > 1: mỗi GPU một tiến trình train.py)
    alias_run(src_cfg, dst_cfg)       -> summary        (dùng lại một lần chạy y hệt cấu hình + seed)
    load_model(run_dir, device)       -> (model, config)
    full_loader(df, images_dir, ...)  -> DataLoader ảnh nguyên 256x256 (không crop) cho TTA
    tta_views(crop)                   -> {tên view: hàm(batch 256) -> batch}
    fit_temperature_views / apply_temperature_views : temperature scaling cho dự đoán gộp nhiều view
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dataset as D
import inference as INF
from model import build_model
from train import CODE_DIR, REPO_DIR, Config, curve_path, run, run_dir, to_overrides


def _summary(cfg: Config) -> dict:
    return json.loads((run_dir(cfg) / "summary.json").read_text())


def _log_line(s: dict) -> str:
    return (f"{s['exp_id']} seed{s['seed']} {s['backbone']} {s.get('desc', '')}: val macro-F1 {s['val_macro_f1']:.4f} "
            f"top-1 {s['val_top1']:.4f} (best ep {s['best_epoch']}, {s['train_time_per_epoch_s']:.0f} s/epoch train)")


def run_many(cfgs: list[Config], n_parallel: int = 1) -> list[dict]:
    """Chạy các cấu hình chưa có summary.json. n_parallel > 1: mỗi cấu hình là một tiến trình
    `python train.py --set ...` gắn với một GPU (CUDA_VISIBLE_DEVICES), log ở <run_dir>/train.log."""
    todo = [c for c in cfgs if not (run_dir(c) / "summary.json").exists()]
    if n_parallel <= 1:
        for c in todo:
            print(_log_line(run(c)), flush=True)
    elif todo:
        slots: queue.Queue = queue.Queue()
        for i in range(n_parallel):
            slots.put(i)

        def work(c: Config) -> tuple[Config, int]:
            slot = slots.get()
            try:
                rd = run_dir(c)
                rd.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, "LAB_REPO_DIR": str(REPO_DIR),
                       "CUDA_VISIBLE_DEVICES": str(slot) if torch.cuda.is_available() else ""}
                with open(rd / "train.log", "w") as log:
                    code = subprocess.run([sys.executable, str(CODE_DIR / "train.py"), "--set", *to_overrides(c)],
                                          stdout=log, stderr=subprocess.STDOUT, env=env).returncode
                print(_log_line(_summary(c)) if code == 0 else f"{c.exp_id} seed{c.seed}: LỖI (mã {code})", flush=True)
                return c, code
            finally:
                slots.put(slot)

        with ThreadPoolExecutor(n_parallel) as ex:
            failed = [c for c, code in ex.map(work, todo) if code != 0]
        for c in failed:
            print(f"--- {run_dir(c)}/train.log (cuối) ---\n" + (run_dir(c) / "train.log").read_text()[-3000:])
        if failed:
            raise RuntimeError(f"{len(failed)} lần chạy lỗi: {[f'{c.exp_id}/seed{c.seed}' for c in failed]}")
    return [_summary(c) for c in cfgs]


def alias_run(src: Config, dst: Config) -> dict:
    """Dùng lại lần chạy `src` cho `dst` khi hai cấu hình giống hệt nhau (chỉ khác exp_id/desc):
    sao chép thư mục run và ảnh đường cong sang tên mới, ghi rõ nguồn trong config/summary."""
    same = {k: v for k, v in asdict(src).items() if k not in ("exp_id", "desc")}
    other = {k: v for k, v in asdict(dst).items() if k not in ("exp_id", "desc")}
    if same != other:
        diff = {k: (same[k], other[k]) for k in same if same[k] != other[k]}
        raise ValueError(f"Không alias được, cấu hình khác nhau: {diff}")
    s_dir, d_dir = run_dir(src), run_dir(dst)
    if not (d_dir / "summary.json").exists():
        shutil.copytree(s_dir, d_dir, dirs_exist_ok=True)
        for name in ("config.json", "summary.json"):
            d = json.loads((d_dir / name).read_text())
            d.update(exp_id=dst.exp_id, desc=dst.desc, alias_of=f"{src.exp_id}/seed{src.seed}",
                     run_dir=str(d_dir), curve=str(curve_path(dst)))
            (d_dir / name).write_text(json.dumps(d, indent=2))
        if curve_path(src).exists():
            curve_path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(curve_path(src), curve_path(dst))
    return _summary(dst)


def load_model(rd: str | Path, device):
    """Dựng lại model từ config.json và nạp best.pt. Trả về (model ở eval mode, config dict)."""
    rd = Path(rd)
    cfg = json.loads((rd / "config.json").read_text())
    model = build_model(cfg["backbone"], pretrained=False, num_classes=D.NUM_CLASSES, head_std=None)
    model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"))
    model = model.to(device).eval()
    if cfg.get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    return model, cfg


def full_loader(df, images_dir, mean, std, batch_size: int = 64, num_workers: int = 2, size: int = 256):
    """Ảnh nguyên size x size (Resize(size) + CenterCrop(size) = không cắt với ảnh 256x256)."""
    tf = D.build_transforms(False, size, mean=mean, std=std, crop_pct=1.0)
    return D.make_loader(df, images_dir, tf, batch_size, train=False, num_workers=num_workers)


def tta_views(crop: int = 224, full: int = 256, sizes=(224, 256, 288, 320)) -> dict:
    """View cho ảnh nguyên `full` x `full`. "center" trùng đúng tiền xử lý val (Resize 256 + CenterCrop 224)."""
    off = int(round((full - crop) / 2.0))
    views = {"center": lambda x: x[..., off:off + crop, off:off + crop]}
    views["center_hflip"] = lambda x: INF.view_hflip(views["center"](x))
    for i in range(5):
        views[f"crop{i}"] = lambda x, i=i: INF.views_multicrop(x, crop)[i]
        views[f"crop{i}_hflip"] = lambda x, i=i: INF.view_hflip(INF.views_multicrop(x, crop)[i])
    for s in sizes:
        views[f"full{s}"] = lambda x, s=s: INF.views_multiscale(x, [s])[0]
    return views


def _probs_t(view_logits, log_t, space: str):
    z = torch.stack([torch.as_tensor(np.asarray(v), dtype=torch.float64) for v in view_logits]) / log_t.exp()
    return F.softmax(z.mean(0), -1) if space == "logit" else F.softmax(z, -1).mean(0)


def fit_temperature_views(view_logits, labels, space: str = "prob") -> float:
    """Một T cực tiểu NLL trên VAL của dự đoán gộp: mean_k softmax(z_k / T) (prob) hoặc softmax(mean_k z_k / T)."""
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.nll_loss(torch.log(_probs_t(view_logits, log_t, space).clamp_min(1e-12)), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature_views(view_logits, T: float, space: str = "prob") -> np.ndarray:
    with torch.no_grad():
        return _probs_t(view_logits, torch.tensor([np.log(T)], dtype=torch.float64), space).numpy()


def with_overrides(cfg: Config, **kw) -> Config:
    return replace(cfg, **kw)

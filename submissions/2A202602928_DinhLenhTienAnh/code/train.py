"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm. Đặt biến môi trường LAB_REPO_DIR (thư mục chứa eval.py) nếu
train.py không nằm trong <repo>/submissions/<...>/code.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.environ.get("LAB_REPO_DIR", CODE_DIR.parents[2]))
sys.path.insert(0, str(CODE_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

import dataset as D  # noqa: E402
from eval import compute_metrics  # noqa: E402
from losses import build_criterion, class_weights, mix_batch, mixed_loss  # noqa: E402
from model import (build_model, count_gmacs, count_params, data_config, param_groups,  # noqa: E402
                   set_train_mode, weight_tag)


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng trong tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    head_std: float | None = 1e-3     # khởi tạo lại head (model.reset_head); None = init của timm
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # none | basic | vflip | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.1      # chỉ dùng khi loss = "ls"
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None   # ce_weighted: None/0 = 1/n_c, > 0 = class-balanced
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    preload: bool = False             # đọc trước file JPEG vào RAM (dataset.DeepWeedsDataset)
    deterministic: bool = False
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"        # ảnh biểu đồ training <exp_id>_<mota>.png (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- tiện ích ---
    resume: bool = True               # đã có summary.json thì không chạy lại
    debug_subset: int | None = None   # CHỈ để thử code: lấy n ảnh mỗi lớp ở mỗi tập (phá S1, không dùng thật)


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    """curves/<exp_id>_<desc>.png cho seed 0; các seed khác thêm _seed<k>."""
    seed = "" if cfg.seed == 0 else f"_seed{cfg.seed}"
    desc = f"_{cfg.desc}" if cfg.desc else ""
    return Path(cfg.curves_dir) / f"{cfg.exp_id}{seed}{desc}.png"


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Cố định random, numpy, torch (CPU và CUDA).

    Worker của DataLoader được seed trong dataset.make_loader (generator + seed_worker).
    deterministic=False giữ cudnn.benchmark=True cho nhanh: cùng seed cho kết quả rất gần nhưng
    không trùng từng bit trên GPU. deterministic=True tái lập chặt hơn nhưng chậm hơn.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (model.param_groups): weight decay không áp dụng cho norm/bias."""
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật THEO BƯỚC. Giữ nguyên tỉ lệ LR giữa các nhóm."""
    total = max(1, cfg.epochs * steps_per_epoch)
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    - bản sao riêng `self.module` (eval mode) dùng để đánh giá
    - decay khởi động d_t = min(d, (1 + t) / (10 + t)) để vài trăm bước đầu không bị trọng số
      ban đầu (head gần 0) kéo lại
    - buffer BatchNorm (running_mean/var) cũng được lấy trung bình động; buffer nguyên (đếm batch) sao chép
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1 - d)
            else:
                v.copy_(src[k])


def _autocast(device, enabled: bool):
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lrs" (LR head theo bước), ...}."""
    set_train_mode(model)  # giữ BN ở eval nếu backbone bị đóng băng
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n, lrs, t0 = 0, [], time.time()
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with _autocast(device, cfg.amp):
            if cfg.mix:
                x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
                logits = model(x)
                loss = mixed_loss(criterion, logits, targets)
            else:
                logits = model(x)
                loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.detach() * len(y)
        if not cfg.mix:  # accuracy trên batch đã trộn không có nghĩa
            correct += (logits.detach().argmax(1) == y).sum()
        n += len(y)
        lrs.append(optimizer.param_groups[-1]["lr"])
    return {"train_loss": loss_sum.item() / n,
            "train_acc": float("nan") if cfg.mix else correct.item() / n,
            "train_time_s": time.time() - t0, "lrs": lrs}


def evaluate(model, loader, criterion, device, amp: bool = True, channels_last: bool = True):
    """Chạy model trên loader ở chế độ eval, KHÔNG tính gradient, giữ đúng thứ tự file.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9], loss: float).
    criterion=None thì loss là cross-entropy thường (so sánh được giữa các loss huấn luyện khác nhau).
    """
    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode(), _autocast(device, amp):
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            outs.append(model(x).float().cpu())
            ys.append(y)
            names.extend(f)
    logits, y = torch.cat(outs), torch.cat(ys)
    loss = (criterion or F.cross_entropy)(logits, y).item()
    return names, y.numpy(), logits.numpy(), loss


def softmax_np(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics_from_logits(y, logits) -> dict:
    probs = softmax_np(logits)
    return compute_metrics(np.asarray(y), probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs: list[float] | None = None) -> None:
    """curves/<exp_id>_<mota>.png: loss train/val, macro-F1 + top-1 val (và acc train), LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    ncols = 3 if lrs else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 3.8))
    axes[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss (loss huấn luyện)")
    axes[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    axes[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    axes[1].plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    if not all(math.isnan(h["train_acc"]) for h in history):
        axes[1].plot(ep, [h["train_acc"] for h in history], "^--", label="train top-1")
    if "val_macro_f1_raw" in history[0]:
        axes[1].plot(ep, [h["val_macro_f1_raw"] for h in history], "x:", label="val macro-F1 (không EMA)")
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    axes[1].axvline(best["epoch"], color="gray", ls="--", lw=0.8)
    axes[1].annotate(f"best {best['val_macro_f1']:.4f} @ep{best['epoch']}", (best["epoch"], best["val_macro_f1"]),
                     textcoords="offset points", xytext=(-60, -18), fontsize=8)
    axes[1].set(xlabel="epoch", ylabel="metric", title="Chỉ số")
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if lrs:
        axes[2].plot(np.arange(1, len(lrs) + 1) / (len(lrs) / len(history)), lrs)
        axes[2].set(xlabel="epoch", ylabel="LR (nhóm head)", title="LR theo bước (warmup + cosine)")
        axes[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def library_versions() -> dict:
    import timm
    import torchvision
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def _subset(df, n):
    return df.groupby("Label", group_keys=False).head(n).reset_index(drop=True) if n else df


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict tóm tắt (cũng ghi summary.json).

    Lưu ở run_dir(cfg): config.json, history.csv (ghi sau mỗi epoch), best.pt (state_dict của epoch có
    macro-F1 val cao nhất, hòa thì epoch sớm hơn; là trọng số EMA nếu bật EMA), val_logits.npz,
    summary.json; và ảnh curves/<exp_id>_<desc>.png.
    KHÔNG dùng test để chọn gì; test chỉ được chạy khi cfg.save_test_predictions (Bước 4).
    """
    import pandas as pd

    rd = run_dir(cfg)
    if cfg.resume and (rd / "summary.json").exists():
        return json.loads((rd / "summary.json").read_text())
    set_seed(cfg.seed, cfg.deterministic)
    rd.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    train_df, val_df, test_df = (_subset(df, cfg.debug_subset) for df in (train_df, val_df, test_df))

    model = build_model(cfg.backbone, pretrained=True, num_classes=D.NUM_CLASSES, drop_rate=cfg.drop_rate,
                        init=cfg.init, head_std=cfg.head_std).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    dc = data_config(model)
    mean, std = dc["mean"], dc["std"]
    info = {"weight_tag": weight_tag(model), "params_M": count_params(model),
            "gmacs": count_gmacs(model, cfg.img_size), "mean": list(mean), "std": list(std),
            "n_train": len(train_df), "n_val": len(val_df), "versions": library_versions()}
    (rd / "config.json").write_text(json.dumps({**asdict(cfg), **info}, indent=2))

    train_loader = D.make_loader(train_df, cfg.images_dir, D.build_transforms(True, cfg.img_size, cfg.aug, mean, std),
                                 cfg.batch_size, train=True, sampler=cfg.sampler, num_workers=cfg.num_workers,
                                 seed=cfg.seed, preload=cfg.preload)
    val_tf = D.build_transforms(False, cfg.img_size, mean=mean, std=std)
    val_loader = D.make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size * 2, train=False,
                               num_workers=cfg.num_workers, preload=cfg.preload)

    weight = None
    if cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"], minlength=D.NUM_CLASSES)   # chỉ số liệu train
        weight = class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weight)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    history, lrs = [], []
    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        lrs += tr.pop("lrs")
        eval_model = ema.module if ema else model
        names, y, logits, val_loss = evaluate(eval_model, val_loader, None, device, cfg.amp, cfg.channels_last)
        m = metrics_from_logits(y, logits)
        row = {"epoch": epoch, **tr, "val_loss": val_loss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_balanced_acc": m["balanced_acc"], "val_ece": m["ece"], "lr_end": lrs[-1],
               "epoch_time_s": time.time() - t_epoch}
        if ema:
            row["val_macro_f1_raw"] = metrics_from_logits(*evaluate(model, val_loader, None, device, cfg.amp,
                                                                    cfg.channels_last)[1:3])["macro_f1"]
        history.append(row)
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        if m["macro_f1"] > best_f1:   # hòa thì giữ epoch sớm hơn
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(eval_model.state_dict(), rd / "best.pt")
            np.savez_compressed(rd / "val_logits.npz", filenames=np.array(names), y=y, logits=logits)
        print(f"[{cfg.exp_id} s{cfg.seed}] ep {epoch:2d}/{cfg.epochs} loss {tr['train_loss']:.4f} "
              f"val_loss {val_loss:.4f} val_F1 {m['macro_f1']:.4f} top1 {m['top1']:.4f} "
              f"({row['epoch_time_s']:.0f}s)", flush=True)

    model.load_state_dict(torch.load(rd / "best.pt", map_location=device))
    vz = np.load(rd / "val_logits.npz")
    mv = metrics_from_logits(vz["y"], vz["logits"])
    if cfg.save_test_predictions:   # Bước 4: test đúng MỘT lần, bằng checkpoint đã chọn trên val
        from eval import save_predictions
        test_loader = D.make_loader(test_df, cfg.images_dir, val_tf, cfg.batch_size * 2, train=False,
                                    num_workers=cfg.num_workers)
        tn, ty, tl, _ = evaluate(model, test_loader, None, device, cfg.amp, cfg.channels_last)
        np.savez_compressed(rd / "test_logits.npz", filenames=np.array(tn), y=ty, logits=tl)
        save_predictions(pred_path(cfg, "test"), tn, ty, softmax_np(tl))

    plot_curves(history, curve_path(cfg), f"{cfg.exp_id} seed {cfg.seed} - {cfg.backbone} {cfg.desc}".strip(), lrs)
    train_times = [h["train_time_s"] for h in history]
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "desc": cfg.desc, **info,
        "best_epoch": best_epoch, "epochs": cfg.epochs, "img_size": cfg.img_size,
        "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"], "val_balanced_acc": mv["balanced_acc"],
        "val_ece": mv["ece"], "val_nll": mv["nll"],
        "val_f1_per_class": dict(zip(D.CLASS_NAMES, map(float, mv["f1"]))),
        "train_time_per_epoch_s": float(np.mean(train_times)),
        "wall_time_total_s": float(sum(h["epoch_time_s"] for h in history)),
        "run_dir": str(rd), "curve": str(curve_path(cfg)),
    }
    if ema:
        summary["val_macro_f1_raw_at_best"] = history[best_epoch - 1]["val_macro_f1_raw"]
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    del model, ema, optimizer, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _cast(value: str, type_str: str):
    if value.lower() in ("none", "null") and "None" in type_str:
        return None
    if "bool" in type_str:
        if value.lower() in ("1", "true", "yes"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"không đọc được bool từ {value!r}")
    if type_str.startswith("int"):
        return int(value)
    if type_str.startswith("float"):
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    types = {f.name: str(f.type) for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"{pair!r} không có dạng KEY=VALUE")
        key, value = pair.split("=", 1)
        if key not in types:
            raise KeyError(f"Config không có trường {key!r}; các trường: {sorted(types)}")
        out[key] = _cast(value, types[key])
    return out


def to_overrides(cfg: Config) -> list[str]:
    """Ngược với parse_overrides: các trường khác mặc định -> ['key=value', ...] (dùng để gọi CLI)."""
    default = Config()
    return [f"{k}={'none' if v is None else v}" for k, v in asdict(cfg).items() if getattr(default, k) != v]


def main(argv=None) -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè trường của Config")
    args = ap.parse_args(argv)
    summary = run(Config(**parse_overrides(args.set)))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()

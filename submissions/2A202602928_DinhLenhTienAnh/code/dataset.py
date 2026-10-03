"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import io
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)

TOTAL_IMAGES = 17509
# Table 1 của bài báo (Olsen et al., 2019), theo thứ tự CLASS_NAMES.
PAPER_COUNTS = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
EXPECTED_RATIO = {"train": 0.6, "val": 0.2, "test": 0.2}
MAX_RATIO_DEV = 0.01  # lệch quá 1 điểm phần trăm so với 60/20/20 thì dừng (README mục 2.1)

# Val/test: Resize(img_size / EVAL_CROP_PCT) rồi CenterCrop(img_size). Với 224: 256 -> 224,
# tức ảnh gốc 256x256 được giữ nguyên rồi cắt giữa 224x224 (GUIDE mục 1.4).
EVAL_CROP_PCT = 0.875
AUG_CHOICES = ("none", "basic", "vflip", "color", "trivial", "randaug")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Trả về ba DataFrame nguyên bản (cột `Filename, Label`), KHÔNG sửa, lọc hay chia lại.
    """
    labels_dir = Path(labels_dir)
    return tuple(pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
                 for split in ("train", "val", "test"))


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, labels_csv: str | Path | None = None,
                verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Sai một ý là raise ngay.

      1. số ảnh mỗi tập, mỗi lớp trong từng tập; tỉ lệ lệch 60/20/20 không quá 1 điểm %
      2. giao từng cặp tập theo Filename phải rỗng; không trùng tên trong cùng một tập
      3. hợp ba tập đúng 17.509 ảnh
      4. mọi Filename tồn tại trong `images_dir`
      5. (nếu có `labels_csv`) đối chiếu nhãn của fold với labels.csv: chỉ cảnh báo, không dừng
    Trả về dict số liệu để dán vào báo cáo.
    """
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(df["Filename"].astype(str)) for k, df in splits.items()}

    for k, df in splits.items():
        if not {"Filename", "Label"} <= set(df.columns):
            raise ValueError(f"{k}: thiếu cột Filename/Label, có {list(df.columns)}")
        if df["Filename"].duplicated().any():
            raise ValueError(f"{k}: có Filename bị trùng trong cùng một tập")
        bad = ~df["Label"].isin(range(NUM_CLASSES))
        if bad.any():
            raise ValueError(f"{k}: nhãn ngoài 0..{NUM_CLASSES - 1}: {df.loc[bad, 'Label'].unique()}")

    # 1. số ảnh và tỉ lệ
    n = {k: len(df) for k, df in splits.items()}
    total = sum(n.values())
    ratio = {k: n[k] / total for k in n}
    for k, r in ratio.items():
        if abs(r - EXPECTED_RATIO[k]) > MAX_RATIO_DEV:
            raise ValueError(f"{k}: tỉ lệ {r:.2%} lệch quá {MAX_RATIO_DEV:.0%} so với "
                             f"{EXPECTED_RATIO[k]:.0%}: báo giảng viên trước khi chạy tiếp")

    per_class = pd.DataFrame(
        {k: df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
         for k, df in splits.items()})
    per_class["total"] = per_class.sum(axis=1)
    per_class["paper"] = PAPER_COUNTS
    per_class.index = CLASS_NAMES
    per_class.index.name = "class"

    # 2. giao từng cặp
    overlap = {f"{a}∩{b}": len(names[a] & names[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlap.values()):
        raise ValueError(f"Giao giữa các tập khác rỗng: {overlap}")

    # 3. hợp ba tập
    union = len(names["train"] | names["val"] | names["test"])
    if union != TOTAL_IMAGES:
        raise ValueError(f"Hợp ba tập có {union} ảnh, kỳ vọng {TOTAL_IMAGES}")

    # 4. file tồn tại
    images_dir = Path(images_dir)
    on_disk = {p.name for p in images_dir.iterdir()}
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)
    if missing:
        raise FileNotFoundError(f"{len(missing)} file trong CSV không có trong {images_dir}, "
                                f"ví dụ {missing[:5]}")

    # 5. nhãn khớp labels.csv. Fold 0 gốc có 1 ảnh lệch (20170714-110407-3.jpg: train_subset0
    # ghi 0, labels.csv ghi 1). Theo S1 vẫn dùng nguyên CSV của fold, chỉ cảnh báo để ghi báo cáo.
    label_mismatch = None
    if labels_csv is not None:
        ref = pd.read_csv(labels_csv).set_index("Filename")["Label"]
        rows = pd.concat([df.assign(split=k) for k, df in splits.items()])
        rows["Label_labels_csv"] = ref.reindex(rows["Filename"]).to_numpy()
        label_mismatch = rows[rows["Label"] != rows["Label_labels_csv"]].reset_index(drop=True)
        if len(label_mismatch):
            warnings.warn(f"{len(label_mismatch)} ảnh có nhãn trong fold khác labels.csv "
                          f"(vẫn dùng nhãn của fold theo S1):\n{label_mismatch.to_string()}")

    stats = {
        "n": {**n, "total": total},
        "ratio": {k: round(r, 4) for k, r in ratio.items()},
        "overlap": overlap,
        "union": union,
        "missing_files": len(missing),
        "label_mismatch": label_mismatch,
        "per_class": per_class,
    }
    if verbose:
        print("Số ảnh:", stats["n"], "| tỉ lệ:", {k: f"{r:.2%}" for k, r in ratio.items()})
        print("Giao từng cặp:", overlap, "| hợp:", union, "| file thiếu:", len(missing),
              "| nhãn lệch labels.csv:", "chưa kiểm" if label_mismatch is None else len(label_mismatch))
        print(per_class.to_string())
        print("OK: các kiểm tra bắt buộc (số ảnh, giao rỗng, hợp 17.509, file tồn tại) đều đạt.")
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = EVAL_CROP_PCT):
    """Tạo transform. `aug` là mức augmentation khi train (trục B của GUIDE.md mục 3):

      - "none"    : giống val (dùng cho kiểm tra overfit một batch)
      - "basic"   : RandomResizedCrop(img_size) + lật ngang          (công thức nền T00)
      - "vflip"   : basic + lật dọc (ảnh chụp từ trên xuống, bài báo xoay ±360° nên hợp lệ)
      - "color"   : basic + ColorJitter
      - "trivial" : basic + TrivialAugmentWide
      - "randaug" : basic + RandAugment(num_ops=2, magnitude=9)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.

    Val/test (train=False, `aug` bị bỏ qua): Resize(img_size / crop_pct) + CenterCrop(img_size).
    Với 224 là Resize(256) + CenterCrop(224); ảnh gốc đã 256x256 nên chỉ cắt giữa.
    """
    normalize = [T.ToTensor(), T.Normalize(mean, std)]
    eval_tf = [T.Resize(round(img_size / crop_pct)), T.CenterCrop(img_size)]
    if not train or aug == "none":
        return T.Compose(eval_tf + normalize)
    if aug not in AUG_CHOICES:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn một trong {AUG_CHOICES}")

    tf = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "vflip":
        tf.append(T.RandomVerticalFlip())
    elif aug == "color":
        tf.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    elif aug == "trivial":
        tf.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        tf.append(T.RandAugment(num_ops=2, magnitude=9))
    return T.Compose(tf + normalize)


def denormalize(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Đảo Normalize để vẽ ảnh: (C, H, W) hoặc (N, C, H, W) -> giá trị trong [0, 1]."""
    m = torch.as_tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=x.dtype).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str).
    `preload=True` đọc trước toàn bộ file JPEG (đã nén, ~30 KB/ảnh) vào một mảng numpy liền khối:
    hết nghẽn đọc đĩa, và worker của DataLoader dùng chung bộ nhớ mà không bị copy-on-write.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None,
                 preload: bool = False):
        self.images_dir = Path(images_dir)
        self.filenames = df["Filename"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.transform = transform
        self._buf = self._offsets = None
        if preload:
            blobs = [(self.images_dir / f).read_bytes() for f in self.filenames]
            self._offsets = np.cumsum([0] + [len(b) for b in blobs])
            self._buf = np.frombuffer(b"".join(blobs), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        if self._buf is not None:
            src = io.BytesIO(self._buf[self._offsets[i]:self._offsets[i + 1]].tobytes())
        else:
            src = self.images_dir / self.filenames[i]
        with Image.open(src) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random trong mỗi worker từ seed torch của worker (tái lập augmentation)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, preload: bool = False, drop_last: bool | None = None):
    """Tạo DataLoader.

      - train=True: shuffle (hoặc sampler); train=False: giữ nguyên thứ tự df để ghép logit với Filename
      - sampler=None | "balanced": "balanced" dùng WeightedRandomSampler, trọng số 1/(số ảnh của lớp)
        (trục D của GUIDE.md mục 3), rút có hoàn lại len(df) mẫu mỗi epoch
      - drop_last mặc định = train (tránh batch cuối quá nhỏ làm BatchNorm không ổn định)
      - `seed` cố định thứ tự batch/sampler; seed_worker cố định augmentation trong worker
    """
    if sampler not in (None, "balanced"):
        raise ValueError(f"sampler={sampler!r} không hợp lệ, chọn None hoặc 'balanced'")
    ds = DeepWeedsDataset(df, images_dir, transform, preload=preload)
    g = torch.Generator()
    g.manual_seed(seed)

    weighted = None
    if train and sampler == "balanced":
        labels = np.asarray(ds.labels)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        weights = 1.0 / counts[labels]
        weighted = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                         num_samples=len(ds), replacement=True, generator=g)

    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=train and weighted is None, sampler=weighted,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=train if drop_last is None else drop_last,
        worker_init_fn=seed_worker, generator=g,
        persistent_workers=num_workers > 0,
    )

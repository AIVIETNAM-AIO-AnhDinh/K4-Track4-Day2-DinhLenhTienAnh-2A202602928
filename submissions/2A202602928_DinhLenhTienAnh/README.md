# Lab Day 2 — DeepWeeds · Đinh Lệnh Tiến Anh (2A202602928)

> Notebook chạy toàn bộ Bước 0–5 trên Kaggle (T4 x2, ~2–3 giờ). Kết quả: `results.xlsx`, `report_draft.md`, `figures/`,
> `curves/`, `predictions/`, `eval_out/` trong tab Output. Các mục còn trống (`…`) điền sau khi chạy.

## Notebook chạy lại

- Kaggle: … (link notebook sau khi *Save Version*)
- File: [`code/lab_day2.ipynb`](code/lab_day2.ipynb)

## Chạy trên Kaggle

### Cách 1: Kaggle CLI từ máy local (khuyên dùng)

Cần `kaggle` CLI (`uv tool install kaggle`) và token ở `~/.kaggle/access_token`.
`code/kernel-metadata.json` đặt sẵn: kernel riêng tư `tienanh211/deepweeds-lab-day2`, GPU T4 x2, Internet bật.

```bash
git push                                                        # notebook clone code từ GitHub
kaggle kernels push -p submissions/2A202602928_DinhLenhTienAnh/code   # chạy cả notebook trên Kaggle
kaggle kernels status tienanh211/deepweeds-lab-day2             # RUNNING -> COMPLETE / ERROR
kaggle kernels output tienanh211/deepweeds-lab-day2 -p /tmp/kaggle_out   # tải Output + log về
```

Rồi chép `figures/`, `curves/`, `predictions/`, `env.json` từ `/tmp/kaggle_out` vào thư mục này (không chép `runs/`).

### Cách 2: giao diện web

1. `git push` code lên GitHub (notebook tự `git clone` repo này, nhánh `main`).
2. Kaggle → *Create → New Notebook* → *File → Import Notebook* → tải lên `code/lab_day2.ipynb`.
3. *Settings*: **Accelerator = GPU T4 x2**, **Internet = On**. P100 có thể không chạy được với bản torch mới, ô cài đặt sẽ báo lỗi sớm nếu gặp.
4. *Run All* để chạy thử; chạy dài thì *Save Version → Save & Run All (Commit)*, tắt trình duyệt vẫn chạy.
5. Kết quả ở `/kaggle/working` (tab *Output*): `figures/`, `runs/`, `predictions/`, `curves/`, `env.json`.
   Tải về rồi chép `figures/`, `curves/`, `predictions/` vào thư mục này. Không commit `runs/` (có checkpoint).

Dữ liệu: notebook tự tải `images.zip` từ Zenodo (kiểm tra MD5) và các file CSV của fold 0 từ GitHub của tác giả,
đặt ở `/kaggle/temp/data` (không lưu vào Output). Nếu gắn sẵn một dataset DeepWeeds vào *Kaggle Input*, notebook sẽ dùng luôn.

Chạy local: mở `code/lab_day2.ipynb` từ trong repo; notebook tự tìm `eval.py`, dùng ảnh ở `<repo>/images` nếu có,
và ghi kết quả vào thư mục này.

## Thứ tự chạy

Toàn bộ trong `code/lab_day2.ipynb`, từ trên xuống: cài đặt và tải dữ liệu → Bước 0 → Bước 1 (backbone) → Bước 2 (công thức)
→ Bước 3 (suy luận) → Bước 4 (chung kết, `eval.py`) → Bước 5 (xlsx, báo cáo).
Mọi thí nghiệm đi qua `train.run(Config(**PATHS, exp_id=..., ...))`.

## Môi trường

Phiên bản thư viện, GPU và commit git của mỗi lần chạy được ghi ở `env.json` (ô cài đặt của notebook).

| | |
|---|---|
| Nền tảng / GPU | … |
| python · torch · torchvision · timm | … |

## Seed

Bước 0–3: seed 0. Chung kết (Bước 4): …

## Ghi chú về dữ liệu (phát hiện ở Bước 0)

- Fold 0 có 10.501 / 3.501 / 3.507 ảnh (59,97% / 20,00% / 20,03%), giao từng cặp rỗng, hợp đủ 17.509 ảnh, không thiếu file.
- Ảnh `20170714-110407-3.jpg` có nhãn 0 (Chinee Apple) trong `train_subset0.csv` nhưng nhãn 1 (Lantana) trong `labels.csv`.
  Theo quy tắc S1 vẫn dùng nguyên nhãn của fold, nên tổng theo lớp lệch Table 1 một ảnh (Chinee Apple 1.126, Lantana 1.063).
  Ảnh này nằm trong train nên không ảnh hưởng tới tập test.
- Head khởi tạo mặc định của timm cho EfficientNet-B0/MobileNetV3 cho loss ban đầu ≈ 5,6–5,9 (xa ln 9 = 2,197).
  `model.build_model` khởi tạo lại head (N(0, 0,001), bias 0) cho mọi backbone để loss ban đầu ≈ ln 9
  (std 0,01 vẫn lệch ~0,1 với Swin/ConvNeXt).

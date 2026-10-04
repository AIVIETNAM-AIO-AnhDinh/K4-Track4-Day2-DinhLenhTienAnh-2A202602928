# Báo cáo Lab Day 2 — DeepWeeds (BẢN NHÁP TỰ SINH)

> Sinh tự động từ lần chạy kaggle (Tesla T4, torch 2.11.0+cu128, timm 1.0.29, commit ead0f1c).
> Mọi số dưới đây lấy từ `results.xlsx` / `eval_out/`. Các mục **TODO** là phần phân tích bạn tự viết.

## 1. Tóm tắt
- Cấu hình cuối: **convnext_tiny** + công thức **T09** + suy luận **I04_288** + temperature scaling.
- macro-F1 test (3 seed): **0.9804 ± 0.0019** so với mốc T00+I00 0.9715 ± 0.0036
  (Δ +0.0090); top-1 test 0.9835 ± 0.0019.
- Recall lớp khó (test, trung bình 3 seed): Chinee apple 97.3%, Snake weed 95.1%.
- TODO: kết luận chính.

## 2. Dữ liệu và thiết lập
- DeepWeeds fold 0: {'train': 10501, 'val': 3501, 'test': 3507, 'total': 17509} ảnh (train/val/test/tổng); giao từng cặp rỗng; hợp 17509; không thiếu file.
  Một ảnh lệch nhãn giữa `train_subset0.csv` và `labels.csv` (xem `figures/split_checks.json`), dùng nguyên nhãn fold theo S1.
- Phân bố lớp: `figures/eda_class_distribution.png`; mất cân bằng train 9.03x (Negatives / lớp nhỏ nhất).
- Kiểm tra pipeline (`figures/pipeline_checks.json`): loss ban đầu ≈ ln 9 cho mọi backbone sau khi khởi tạo lại head;
  overfit 1 batch tới loss 0.0009; BN eval/train được kiểm tra.
- Công thức nền T00: AdamW, LR backbone 1e-4 / head 1e-3, wd 0,05 (không cho norm/bias), warmup 1 epoch + cosine,
  CE, batch 64, 12 epoch, AMP, RandomResizedCrop(224) + lật ngang; val/test: Resize 256 + CenterCrop 224.
- Chọn checkpoint: epoch có macro-F1 val cao nhất. Seed: 0 cho Bước 1-3; 0, 1, 2 cho chung kết.

## 3. So sánh backbone
| exp_id | backbone | weight_tag | params_M | GMAC | val_macro_f1 | val_top1 | train_s_per_epoch | latency_b1_fp32_p50_ms | note |
|---|---|---|---|---|---|---|---|---|---|
| B01 | resnet50 | resnet50.a1_in1k | 23.5265 | 4.0872 | 0.8025 | 0.8580 | 33.9608 | 6.1407 |  |
| B02 | convnext_tiny | convnext_tiny.in12k_ft_in1k | 27.8270 | 4.4548 | 0.9681 | 0.9743 | 50.3780 | 5.5573 | CHỌN đi tiếp |
| B03 | deit_small_patch16_224 | deit_small_patch16_224.fb_in1k | 21.6691 | 4.5985 | 0.9492 | 0.9640 | 32.8700 | 5.2111 |  |
| B04 | swin_tiny_patch4_window7_224 | swin_tiny_patch4_window7_224.ms_in1k | 27.5263 | 4.4898 | 0.9542 | 0.9674 | 61.6991 | 9.7591 |  |
| B05 | efficientnet_b0 | efficientnet_b0.ra_in1k | 4.0191 | 0.3845 | 0.8581 | 0.8920 | 33.0253 | 7.8100 |  |
| B06 | mobilenetv3_large_100 | mobilenetv3_large_100.ra_in1k | 4.2136 | 0.2153 | 0.7820 | 0.8358 | 23.2573 | 6.2615 |  |

Chọn: B02 convnext_tiny: macro-F1 val 0.9681 (cao nhất 0.9681 là B02 convnext_tiny); trong 1 backbone cách mức cao nhất ≤ 0.01, đây là cái có độ trễ batch-1 thấp nhất (5.56 ms FP32).

TODO: nhận xét (hội tụ, quá khớp, thứ hạng so với ImageNet, FLOPs vs độ trễ). Hình: `figures/backbones_f1_latency.png`.

## 4. Công thức huấn luyện (convnext_tiny, 1 seed)
Nhiễu không tất định (T00 vs B02, cùng cấu hình + seed): 0.0000; ngưỡng thắng MIN_GAIN = 0.0050. Cách làm: tham lam theo trục.

| exp_id | axis | change_vs_T00 | val_macro_f1 | val_top1 | delta_vs_T00 | F1_ChineeApple | F1_SnakeWeed | note |
|---|---|---|---|---|---|---|---|---|
| T00 | nền | công thức nền | 0.9681 | 0.9743 | 0.0000 | 0.9490 | 0.9369 |  |
| T01 | A | init=frozen (chỉ train head) | 0.8534 | 0.8837 | -0.1148 | 0.8177 | 0.7681 |  |
| T02 | A | init=scratch (từ đầu) | 0.2928 | 0.5538 | -0.6754 | 0.1887 | 0.2857 |  |
| T03 | B | aug=TrivialAugmentWide | 0.9706 | 0.9783 | 0.0025 | 0.9571 | 0.9307 |  |
| T04 | B | CutMix alpha=1 | 0.9698 | 0.9763 | 0.0017 | 0.9543 | 0.9129 |  |
| T05 | B | thêm lật dọc | 0.9665 | 0.9737 | -0.0016 | 0.9490 | 0.9346 |  |
| T06 | C | label smoothing 0.1 | 0.9694 | 0.9763 | 0.0012 | 0.9349 | 0.9294 |  |
| T07 | C | focal loss gamma=2 | 0.9702 | 0.9763 | 0.0020 | 0.9548 | 0.9392 |  |
| T08 | C | CE trọng số 1/n_c | 0.9700 | 0.9769 | 0.0019 | 0.9524 | 0.9320 |  |
| T09 | D | sampler cân bằng lớp | 0.9727 | 0.9780 | 0.0045 | 0.9577 | 0.9420 | CHỌN (cấu hình cuối);  |
| T10 | F | EMA decay 0.999 | 0.9674 | 0.9740 | -0.0008 | 0.9464 | 0.9346 | EMA, F1 không EMA 0.9679 |
| T11 | E | LR head = LR backbone = 1e-4 | 0.9669 | 0.9749 | -0.0012 | 0.9478 | 0.9227 |  |

Cấu hình cuối: T09 (sampler cân bằng lớp): macro-F1 val 0.9727 cao nhất trong các lần chạy T. TODO: yếu tố nào giúp/không giúp và vì sao. Hình: `figures/training_ablation.png`.

## 5. Suy luận (T09 seed 0, val)
| exp_id | method | K | val_macro_f1 | val_top1 | val_ece | p50_ms | p95_ms | relative_cost_vs_I00 | note |
|---|---|---|---|---|---|---|---|---|---|
| I00 | 1 view (center crop 224) | 1.0000 | 0.9727 | 0.9780 | 0.0111 | 7.8944 | 12.1981 | 1.0000 |  |
| I01 | TTA lật ngang | 2.0000 | 0.9723 | 0.9783 | 0.0088 | 7.7524 | 8.1509 | 0.9820 |  |
| I02a | TTA 5 crop | 5.0000 | 0.9718 | 0.9774 | 0.0093 | 10.3326 | 10.4915 | 1.3088 |  |
| I02b | TTA 5 crop + lật (10 view) | 10.0000 | 0.9731 | 0.9789 | 0.0088 | 17.0055 | 17.2857 | 2.1541 |  |
| I03a | TTA lật ngang, gộp LOGIT | 2.0000 | 0.9723 | 0.9783 | 0.0118 | 7.8857 | 8.4788 | 0.9989 |  |
| I03b | TTA 10 view, gộp LOGIT | 10.0000 | 0.9736 | 0.9791 | 0.0108 | 17.1215 | 17.4422 | 2.1688 |  |
| I04_224 | ảnh nguyên resize về 224 | 1.0000 | 0.9773 | 0.9829 | 0.0080 | 8.0279 | 11.1470 | 1.0169 |  |
| I04_256 | ảnh nguyên 256 | 1.0000 | 0.9701 | 0.9760 | 0.0113 | 7.8794 | 8.3352 | 0.9981 |  |
| I04_288 | ảnh nguyên phóng 288 | 1.0000 | 0.9789 | 0.9840 | 0.0070 | 7.9899 | 8.5975 | 1.0121 |  |
| I04_320 | ảnh nguyên phóng 320 | 1.0000 | 0.9781 | 0.9834 | 0.0054 | 7.9035 | 8.3306 | 1.0011 |  |
| I05 | ensemble 3 backbone (trung bình xác suất) | 3.0000 | 0.9700 | 0.9791 | 0.0128 | 27.2736 | 28.9861 | 3.4548 |  |
| I06 | trọng số EMA vs không EMA (T10) | 1.0000 | 0.9674 | 0.9740 | 0.0098 | 7.8944 | 12.1981 | 1.0000 | không EMA cùng epoch: macro-F1 0.9679; EMA không tốn thêm lúc suy luận |
| I07 | 1 view + temperature scaling (T = 1.361) | 1.0000 | 0.9727 | 0.9780 | 0.0062 | 7.8944 | 12.1981 | 1.0000 | ECE val 0.0111 → 0.0062, NLL 0.0992 → 0.0903 (khớp và đo cùng trên val) |
| I08 | FP32 | 1.0000 | 0.9727 | 0.9780 | 0.0103 | 5.5811 | 5.9362 | 0.7070 | lệch logit lớn nhất so với FP32: 0.00e+00 |
| I08 | FP16 (model.half) | 1.0000 | 0.9727 | 0.9780 | 0.0113 | 5.4505 | 5.8105 | 0.6904 | lệch logit lớn nhất so với FP32: 6.53e-02 |
| I08 | gộp BN |  |  |  |  |  |  |  | không áp dụng: backbone không có BatchNorm (0) |

Phương pháp cuối: I04_288 (p95 batch-1 8.60 ms). TODO: đánh đổi ngoại tuyến vs thời gian thực. Hình: `figures/inference_tradeoff.png`.

## 6. Chung kết (test, mỗi seed chạy một lần)
| exp_id | config | seed | val_macro_f1 | test_macro_f1 | test_top1 | test_balanced_acc | test_ece |
|---|---|---|---|---|---|---|---|
| F01 | convnext_tiny + T09 + I04_288 + TS | 0 | 0.9789214649133362 | 0.9803922921817462 | 0.9837467921300256 | 0.9814190712264348 | 0.0050301006244653 |
| F01 | convnext_tiny + T09 + I04_288 + TS | 1 | 0.9770592551326018 | 0.982384160387538 | 0.98517251211862 | 0.9813389152255876 | 0.0029948756829199 |
| F01 | convnext_tiny + T09 + I04_288 + TS | 2 | 0.9755953881849229 | 0.9785502909845148 | 0.9814656401482748 | 0.9756508383069016 | 0.006982748360422 |
| F01 | convnext_tiny + T09 + I04_288 + TS | mean ± std (3 seed, ddof=1) | 0.9772 ± 0.0017 | 0.9804 ± 0.0019 | 0.9835 ± 0.0019 | 0.9795 ± 0.0033 | 0.0050 ± 0.0020 |
| T00 | convnext_tiny + T00 + I00 (mốc) | 0 | 0.9681417560025941 | 0.9685265882274354 | 0.9743370402053037 | 0.9727272561442613 | 0.0088682376247504 |
| T00 | convnext_tiny + T00 + I00 (mốc) | 1 | 0.9685225592178388 | 0.9754574385407512 | 0.9797547761619618 | 0.9809530117752678 | 0.0081047280011405 |
| T00 | convnext_tiny + T00 + I00 (mốc) | 2 | 0.9704935093885843 | 0.9704558586741836 | 0.9769033361847732 | 0.9714845020401512 | 0.010153634128885 |
| T00 | convnext_tiny + T00 + I00 (mốc) | mean ± std (3 seed, ddof=1) | 0.9691 ± 0.0013 | 0.9715 ± 0.0036 | 0.9770 ± 0.0027 | 0.9751 ± 0.0051 | 0.0090 ± 0.0010 |
| F01_uncal | F01 chưa temperature scaling | mean ± std |  |  |  |  | 0.0074 ± 0.0020 |

Tự chấm phần I (`eval.py grade`): 19 / 20.
| code | criterion | points | max | note |
|---|---|---|---|---|
| I1 | Top-1 accuracy test | 7 | 7 | 98.35% (mean 3 seed) |
| I2 | Macro-F1 cải thiện so với mốc | 4 | 5 | final 0.9804, mốc 0.9715, Δ=+0.0090, s=0.0036 |
| I3 | Recall hai lớp khó | 4 | 4 | Chinee Apple 97.3% (mốc 88.5%), Snake Weed 95.1% (mốc 88.8%) |
| I4a | ECE sau TS < ECE trước | 1 | 1 | trước 0.0074, sau 0.0050 |
| I4b | Chênh macro-F1 val/test <= 0.02 | 1 | 1 | val 0.9772, test 0.9804, chênh 0.0033 |
| I5 | Cấu hình thời gian thực | 2 | 2 | p95 = 8.6 ms (ngân sách 100 ms), đo đúng cách |

Ma trận nhầm lẫn: `figures/final_confusion_test.png`; ảnh sai: `figures/final_errors_test.jpg`.
TODO: lớp nào còn nhầm nhiều nhất và giả thuyết nguyên nhân.

## 7. Kết luận và khuyến nghị
TODO: cấu hình tốt nhất, yếu tố đóng góp nhiều nhất, lựa chọn cho robot 30–100 ms/khung.

## 8. Hạn chế
- Một fold, chia ngẫu nhiên (không theo địa điểm); ablation 1 seed; 12 epoch (bài báo ~100 epoch).
- TODO.

## 9. Phụ lục
Cấu hình đầy đủ: `runs/<exp_id>/seed<k>/config.json`; lựa chọn: `final_choice.json`; môi trường: `env.json`.

# Adaptive Information Routing for Medical Image Segmentation

Triển khai thực nghiệm cho luận văn thạc sĩ HUTECH, xây dựng trên nền TransAttUnet (Chen et al., 2022).

## Đề xuất

TransAttUnet hợp nhất TSA và GSA theo cách **cố định** (`F = F_tsa + F_gsa`), và truyền đặc trưng đa tỉ lệ qua skip connection với **trọng số bằng nhau** bất kể nội dung ảnh. Luận văn đề xuất hai thành phần thay thế:

**ADAR - Adaptive Dual-path Attention Routing** (tại bottleneck)

```
[w_tsa, w_gsa, w_orig] = Softmax(Gate(F_enc))
F_out = w_tsa·F_tsa + w_gsa·F_gsa + w_orig·F_enc
```

**CSR - Cross-scale Semantic Routing** (tại mỗi tầng decoder)

```
[w_prev, w_cur] = Softmax(Gate(cat(F_prev, F_cur)))
F_out = cat(w_prev·F_prev, w_cur·F_cur)
```

## Cấu trúc

```
models/
  backbone.py       # UNet building blocks
  adar.py           # ADAR + FixedSAA (baseline)
  csr.py            # CSR
  transattunet.py   # TransAttUnet_R - baseline
  proposed.py       # Mô hình đề xuất (flags use_adar / use_csr)
datasets/
  isic.py           # ISIC-2018 skin lesion, 256×256
  glas.py           # GlaS gland, 128×128
  covid.py          # Clean-CC-CCII COVID-19, 512×512
  lung.py           # Montgomery CXR lung, 256×256
  dsb2018.py        # Data Science Bowl 2018 nucleus, 256×256
utils/
  metrics.py        # Dice, IoU, ACC, REC, PRE, HD95
  visualize.py      # Segmentation grid, routing weight plots
train.py            # Training loop
evaluate.py         # Evaluation + xuất CSV và visualizations
ablation.py         # Chạy 4 cấu hình ablation, in bảng tổng hợp
download_datasets.py # Tự động tải và chuẩn bị datasets
experiments_kaggle.ipynb  # Notebook chạy trên Kaggle
experiments_colab.ipynb   # Notebook chạy trên Google Colab Pro
```

## Datasets

| Dataset        | Modality      | Split            | Input   | Tải tự động    |
| -------------- | ------------- | ---------------- | ------- | -------------- |
| ISIC-2018      | Dermoscopy    | 2076 / 207 / 519 | 256×256 | => ISIC S3     |
| GlaS           | H&E histology | 85 / 8 / 80      | 128×128 | => Kaggle CLI  |
| Clean-CC-CCII  | Chest CT      | 180 / 20 / 60    | 512×512 | => CNCB public |
| Montgomery CXR | Chest X-ray   | ~96 / ~10 / ~32  | 256×256 | => NLM public  |
| DSB-2018       | Microscopy    | 537 / 67 / 67    | 256×256 | => Kaggle CLI  |

## Cài đặt

```bash
pip install torch torchvision scipy matplotlib
```

## Chuẩn bị dữ liệu

```bash
# Tải tất cả
python download_datasets.py --all --data_dir data/

# Tải từng dataset
python download_datasets.py --isic    --data_dir data/
python download_datasets.py --glas    --data_dir data/          # cần kaggle.json
python download_datasets.py --covid   --data_dir data/
python download_datasets.py --lung    --data_dir data/          # tự động từ NLM
python download_datasets.py --dsb2018 --data_dir data/          # cần accept rules Kaggle

# Tải thủ công nếu cần
python download_datasets.py --glas    --glas_zip    /path/to/warwick.zip
python download_datasets.py --lung    --lung_zip    /path/to/NLM-MontgomeryCXRSet.zip
python download_datasets.py --dsb2018 --dsb2018_zip /path/to/stage1_train.zip
```

> **Kaggle CLI:** Đặt `~/.kaggle/kaggle.json` trước khi tải GlaS và DSB-2018.  
> **DSB-2018:** Cần accept rules tại kaggle.com/competitions/data-science-bowl-2018/rules.  
> **Montgomery:** Tải thủ công tại openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip

## Chạy trên Kaggle

Mở [experiments_kaggle.ipynb](experiments_kaggle.ipynb) - **không cần tài khoản Pro**, GPU T4 miễn phí 30h/tuần.

**Yêu cầu:**

1. Upload thư mục `data/` lên Kaggle Datasets (một lần)
2. Tạo notebook mới => **Add Data** => chọn dataset vừa upload
3. Điền `GITHUB_REPO` và `KAGGLE_DATASET_SLUG` ở cell cấu hình
4. Chạy tuần tự từ trên xuống

| Section | Tác dụng                                     |
| ------- | -------------------------------------------- |
| 0       | Cấu hình dataset, epochs, batch size         |
| 1       | Clone repo từ GitHub, install deps           |
| 2       | Load dataset từ `/kaggle/input/`             |
| 3       | Kiểm tra forward pass các model              |
| 4       | Train 4 cấu hình ablation                    |
| 5       | Evaluate + xuất CSV                          |
| 6       | In bảng ablation                             |
| 7       | Segmentation grid, ADAR weights, CSR weights |
| 8       | Zip kết quả => `/kaggle/working/`            |

**Thời gian ước tính trên T4** (100 epochs):

| Dataset    | Mỗi model | 4 models |
| ---------- | --------- | -------- |
| GlaS       | ~20 phút  | ~1.5 giờ |
| ISIC       | ~70 phút  | ~5 giờ   |
| DSB-2018   | ~60 phút  | ~4 giờ   |
| Montgomery | ~15 phút  | ~1 giờ   |
| COVID      | ~120 phút | ~8 giờ   |

## Chạy trên Google Colab Pro

Mở [experiments_colab.ipynb](experiments_colab.ipynb) => `Runtime` => `Change runtime type` => **GPU: A100**.

## Sử dụng local

```bash
# Train một mô hình
python train.py --dataset isic --data_root data/isic --model proposed

# Đánh giá
python evaluate.py --dataset isic --data_root data/isic \
  --model proposed --checkpoint checkpoints/isic/proposed/best_model.pth

# Chạy toàn bộ ablation study (train + evaluate + bảng kết quả)
python ablation.py --dataset glas --data_root data/glas

# Chỉ evaluate (đã có checkpoint)
python ablation.py --dataset glas --data_root data/glas --skip_train
```

`--model` nhận: `baseline` | `adar_only` | `csr_only` | `proposed`  
`--dataset` nhận: `isic` | `glas` | `covid` | `lung` | `dsb2018`

## Cấu hình training

Nhất quán với paper gốc (TransAttUnet, Bảng I–V):

| Tham số         | Giá trị                              |
| --------------- | ------------------------------------ |
| Optimizer       | SGD, momentum=0.9, weight_decay=1e-4 |
| Learning rate   | 1e-4, ×0.1 tại epoch 40 và 80        |
| Epochs          | 100                                  |
| Batch size      | 4                                    |
| Loss            | 0.5·BCE + 0.5·Dice                   |
| Ngưỡng nhị phân | 0.5                                  |

## Ablation

| Cấu hình    | ADAR | CSR |
| ----------- | ---- | --- |
| `baseline`  | ✗    | ✗   |
| `adar_only` | ✓    | ✗   |
| `csr_only`  | ✗    | ✓   |
| `proposed`  | ✓    | ✓   |

Kết quả lưu tại `results/<dataset>/ablation/ablation_table.txt`.

## Tham khảo

```bibtex
@article{chen2021transattunet,
  title   = {TransAttUnet: Multi-level Attention-guided U-Net with Transformer
             for Medical Image Segmentation},
  author  = {Chen, Bingzhi and Liu, Yishu and Zhang, Zheng and
             Lu, Guangming and Kong, Adams Wai Kin},
  journal = {IEEE Transactions on Instrumentation \& Measurement},
  year    = {2022}
}
```

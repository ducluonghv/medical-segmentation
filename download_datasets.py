"""
Auto-download và chuẩn bị datasets cho thesis experiments.

Tình trạng tự động hóa:
  ISIC-2018   =>  Tải thẳng từ ISIC S3 (public, không cần tài khoản)
  GlaS        =>  Tải từ Kaggle API  (cần kaggle.json)  hoặc --glas_zip thủ công
  COVID       =>  Tải thẳng từ CNCB (public, không cần tài khoản)
              Dataset: CC-CCII  -  https://download.cncb.ac.cn/covid-ct/
  Lung        =>  Tải từ Kaggle API  (cần kaggle.json)  hoặc --lung_zip thủ công
              Dataset: nikhilpandey360/chest-xray-masks-and-labels (JSRT+Montgomery)
  DSB-2018    =>  Tải từ Kaggle competitions  (cần kaggle.json)
              Dataset: data-science-bowl-2018

Kết quả sau khi chạy:
  data/
    isic/
      images/   *.jpg   (2596 ảnh)
      masks/    *_segmentation.png
    glas/
      train/
        images/   train_*.bmp
        masks/    train_*_anno.bmp
      test/
        images/   testA_*.bmp  + testB_*.bmp
        masks/    testA_*_anno.bmp + testB_*_anno.bmp
    covid/
      images/   *.png   (CT slices chứa tổn thương theo lesions_slices.csv)
      masks/    *.png   (binary lesion masks - cần tạo riêng, xem ghi chú)
    lung/
      train/
        images/   *.png  (407 chest X-ray)
        masks/    *.png  (binary lung masks)
      test/
        images/   *.png  (178 ảnh)
        masks/    *.png
    dsb2018/
      images/   *.png   (671 ảnh kính hiển vi, đã flatten)
      masks/    *.png   (binary nucleus masks, đã merge)

Sử dụng:
  python download_datasets.py --all --data_dir data/
  python download_datasets.py --isic --data_dir data/
  python download_datasets.py --glas --data_dir data/
  python download_datasets.py --glas --glas_zip /path/to/warwick_qu_dataset.zip
  python download_datasets.py --covid --data_dir data/
  python download_datasets.py --covid --covid_parts 1 2 3 --data_dir data/  # chỉ tải 1 số zip
  python download_datasets.py --lung --data_dir data/
  python download_datasets.py --lung --lung_zip /path/to/lung_dataset.zip
  python download_datasets.py --dsb2018 --data_dir data/
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


# ──────────────────────────────────────────────────────────────────────
# ISIC-2018
# ──────────────────────────────────────────────────────────────────────

ISIC_FILES = [
    (
        "ISIC2018_Task1-2_Training_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
        "training images",
    ),
    (
        "ISIC2018_Task1_Training_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
        "training masks",
    ),
    (
        "ISIC2018_Task1-2_Test_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip",
        "test images",
    ),
    (
        "ISIC2018_Task1_Test_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Test_GroundTruth.zip",
        "test masks",
    ),
    (
        "ISIC2018_Task1-2_Validation_Input.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
        "validation images",
    ),
    (
        "ISIC2018_Task1_Validation_GroundTruth.zip",
        "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
        "validation masks",
    ),
]


def download_isic(data_dir: Path):
    print("\n" + "=" * 56)
    print("  ISIC-2018 Skin Lesion Segmentation")
    print("=" * 56)

    isic_dir = data_dir / "isic"
    img_dir  = isic_dir / "images"
    msk_dir  = isic_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = isic_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for filename, url, desc in ISIC_FILES:
        zip_path = tmp_dir / filename
        if zip_path.exists():
            print(f"  [skip] {filename} đã tồn tại")
        else:
            print(f"  Đang tải {desc} ...", end="", flush=True)
            _download_with_progress(url, zip_path)
            print(" xong")

        print(f"  Đang giải nén {filename} ...", end="", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if not name:
                    continue
                lower = name.lower()
                src = zf.open(member)
                if "groundtruth" in filename.lower() or "segmentation" in lower:
                    dst = msk_dir / name
                else:
                    if lower.endswith((".jpg", ".jpeg", ".png")):
                        dst = img_dir / name
                    else:
                        continue
                if not dst.exists():
                    with open(dst, "wb") as f:
                        f.write(src.read())
        print(" xong")

    n_imgs  = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    n_masks = len(list(msk_dir.glob("*.png")))
    print(f"\n  ✓ ISIC-2018 sẵn sàng: {n_imgs} ảnh | {n_masks} masks")
    print(f"    => {isic_dir}")

    shutil.rmtree(tmp_dir)


# ──────────────────────────────────────────────────────────────────────
# GlaS
# ──────────────────────────────────────────────────────────────────────

def download_glas(data_dir: Path, glas_zip: str = None):
    print("\n" + "=" * 56)
    print("  GlaS - Gland Segmentation (MICCAI 2015)")
    print("=" * 56)

    glas_dir = data_dir / "glas"

    # Nếu user cung cấp sẵn zip thủ công
    if glas_zip:
        zip_path = Path(glas_zip)
        if not zip_path.exists():
            print(f"  [lỗi] Không tìm thấy file: {glas_zip}")
            _print_glas_instructions()
            return
        _extract_glas(zip_path, glas_dir)
        return

    # Thử Kaggle API
    if shutil.which("kaggle"):
        print("  Phát hiện Kaggle CLI. Đang tải từ Kaggle ...", end="", flush=True)
        tmp = data_dir / "_glas_kaggle"
        tmp.mkdir(exist_ok=True)
        ret = os.system(
            f'kaggle datasets download -d "sani84/glasmiccai2015-gland-segmentation" '
            f'-p "{tmp}" --unzip -q'
        )
        if ret == 0:
            print(" xong")
            _organize_glas_from_flat(tmp, glas_dir)
            shutil.rmtree(tmp, ignore_errors=True)
            return
        else:
            print(" thất bại")

    _print_glas_instructions()


def _organize_glas_from_flat(src_dir: Path, glas_dir: Path):
    """Tổ chức file từ flat directory => train/ và test/ theo cấu trúc project."""
    for split_prefix, dst_split in [("train", "train"), ("testA", "test"), ("testB", "test")]:
        img_out  = glas_dir / dst_split / "images"
        mask_out = glas_dir / dst_split / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for f in sorted(src_dir.rglob(f"{split_prefix}_*.bmp")):
            if "_anno" in f.stem:
                shutil.copy2(f, mask_out / f.name)
            else:
                shutil.copy2(f, img_out / f.name)

    _verify_glas(glas_dir)


def _extract_glas(zip_path: Path, glas_dir: Path):
    print(f"  Đang giải nén {zip_path.name} ...", end="", flush=True)
    tmp = glas_dir.parent / "_glas_tmp"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    print(" xong")
    _organize_glas_from_flat(tmp, glas_dir)
    shutil.rmtree(tmp, ignore_errors=True)


def _verify_glas(glas_dir: Path):
    n_train_img  = len(list((glas_dir / "train" / "images").glob("*.bmp")))
    n_train_msk  = len(list((glas_dir / "train" / "masks").glob("*.bmp")))
    n_test_img   = len(list((glas_dir / "test"  / "images").glob("*.bmp")))
    n_test_msk   = len(list((glas_dir / "test"  / "masks").glob("*.bmp")))
    print(f"\n  ✓ GlaS sẵn sàng:")
    print(f"    train: {n_train_img} ảnh | {n_train_msk} masks")
    print(f"    test : {n_test_img} ảnh | {n_test_msk} masks")
    print(f"    => {glas_dir}")


def _print_glas_instructions():
    print("""
  [!] Không thể tự động tải GlaS. Các cách tải thủ công:

  Cách 1 - Kaggle (khuyến nghị):
    1. Đăng nhập kaggle.com => Account => Create API Token => tải kaggle.json
    2. Đặt vào ~/.kaggle/kaggle.json
    3. Chạy lại script này (sẽ tự động dùng Kaggle CLI)

  Cách 2 - Tải thủ công từ Warwick:
    1. Vào https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest
    2. Tải file zip dataset về máy
    3. Chạy:
       python download_datasets.py --glas --glas_zip /path/to/warwick_dataset.zip
""")


# ──────────────────────────────────────────────────────────────────────
# COVID - CC-CCII (download.cncb.ac.cn, public)
# ──────────────────────────────────────────────────────────────────────

_CNCB_BASE = "https://download.cncb.ac.cn/covid-ct"
_COVID_N_ZIPS = 31   # COVID19-1.zip … COVID19-31.zip


def download_covid(data_dir: Path, parts: "list[int] | None" = None):
    """
    Tải CC-CCII COVID-19 CT scans từ CNCB (public, không cần tài khoản).

    parts: danh sách số thứ tự zip cần tải (mặc định: 1..31).
           Dùng để tải thử hoặc chia nhỏ khi kết nối không ổn định.
    """
    print("\n" + "=" * 56)
    print("  CC-CCII - COVID-19 Pneumonia CT Segmentation")
    print("=" * 56)

    covid_dir = data_dir / "covid"
    img_dir   = covid_dir / "images"
    msk_dir   = covid_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    if img_dir.exists() and len(list(img_dir.glob("*.png"))) > 0:
        n = len(list(img_dir.glob("*.png")))
        print(f"  ✓ COVID đã có {n} ảnh tại {covid_dir}")
        return

    tmp = covid_dir / "_tmp"
    tmp.mkdir(exist_ok=True)

    # 1. Tải lesions_slices.csv
    csv_dst = tmp / "lesions_slices.csv"
    if not csv_dst.exists():
        print("  Tải lesions_slices.csv ...", end="", flush=True)
        _download_with_progress(f"{_CNCB_BASE}/lesions_slices.csv", csv_dst)
        print(" xong")

    # 2. Tải từng COVID19-N.zip
    zip_indices = parts if parts else list(range(1, _COVID_N_ZIPS + 1))
    for i in zip_indices:
        fname    = f"COVID19-{i}.zip"
        zip_dst  = tmp / fname
        if zip_dst.exists():
            print(f"  [skip] {fname} đã tồn tại")
        else:
            print(f"  Tải {fname} ({i}/{len(zip_indices)}) ...", end="", flush=True)
            _download_with_progress(f"{_CNCB_BASE}/{fname}", zip_dst)
            print(" xong")

    # 3. Giải nén và lọc slice theo lesions_slices.csv
    prepare_covid_from_zips(
        covid_zips=[str(tmp / f"COVID19-{i}.zip") for i in zip_indices],
        data_dir=data_dir,
        lesions_csv=str(csv_dst),
    )

    shutil.rmtree(tmp, ignore_errors=True)

    print("""
  Ghi chú về masks:
    CC-CCII KHÔNG kèm pixel-level segmentation masks.
    Sau khi có images/, tạo masks bằng một trong hai cách:
      (a) SAM / GrabCut tự động (nhanh, chất lượng thấp hơn)
      (b) Annotations thủ công / từ advisor
""")


def prepare_covid_from_zips(covid_zips: list, data_dir: Path, lesions_csv: "str | None" = None):
    """
    Giải nén các COVID19-*.zip từ CC-CCII vào cấu trúc chuẩn.

    Cấu trúc bên trong mỗi zip (nested):
      COVID19-N/
        <patient_id>/
          <scan_id>/
            <slice_index>.png   (CT slice, uint16 PNG)

    Nếu có lesions_csv: chỉ giữ lại các slice được liệt kê là có tổn thương.
    Nếu không có      : giữ toàn bộ slice từ các COVID19 zips.
    """
    import csv

    covid_dir = data_dir / "covid"
    img_dir   = covid_dir / "images"
    msk_dir   = covid_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    # Đọc danh sách slice có tổn thương từ lesions_slices.csv nếu có.
    # Cột thường gặp: patient_id, scan_id, slice_index (hoặc filename)
    lesion_keys: set = set()
    if lesions_csv and Path(lesions_csv).exists():
        with open(lesions_csv, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                # Thử build key theo (patient_id, scan_id, slice_index)
                key_parts = [row.get(c, "").strip()
                             for c in ("patient_id", "scan_id", "slice_index")
                             if c in fieldnames]
                if key_parts:
                    lesion_keys.add(tuple(key_parts))
        print(f"  Đã đọc {len(lesion_keys)} slice có tổn thương từ {lesions_csv}")
    else:
        print("  Không có lesions_csv - giữ toàn bộ slice từ COVID19 zips")

    extract_tmp = covid_dir / "_extract"
    extract_tmp.mkdir(exist_ok=True)
    copied = 0

    for zip_path_str in covid_zips:
        zip_path = Path(zip_path_str)
        if not zip_path.exists():
            print(f"  [bỏ qua] Không tìm thấy: {zip_path}")
            continue

        print(f"  Giải nén {zip_path.name} ...", end="", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_tmp)
        print(" xong")

    # Duyệt toàn bộ PNG trong extract_tmp, lọc theo lesion_keys nếu có
    for f in sorted(extract_tmp.rglob("*.png")):
        # parts = [..., patient_id, scan_id, slice_filename]
        parts = f.parts
        if len(parts) >= 3:
            patient_id  = parts[-3]
            scan_id     = parts[-2]
            slice_index = f.stem          # tên file không có .png
            key = (patient_id, scan_id, slice_index)
        else:
            key = None

        # Bỏ qua nếu có CSV và slice này không có trong danh sách
        if lesion_keys and key not in lesion_keys:
            continue

        # Đặt tên file: patient_scan_slice.png để tránh xung đột
        out_name = f"{parts[-3]}_{parts[-2]}_{f.name}" if len(parts) >= 3 else f.name
        dst = img_dir / out_name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied += 1

    shutil.rmtree(extract_tmp, ignore_errors=True)

    n_masks = len(list(msk_dir.glob("*.png")))
    print(f"\n  ✓ COVID images: {copied} slice  =>  {img_dir}")
    if n_masks == 0:
        print("  ⚠ Masks: chưa có - xem ghi chú trong --covid (cần tạo hoặc tải riêng)")
    else:
        print(f"  ✓ COVID masks : {n_masks}  =>  {msk_dir}")


# ──────────────────────────────────────────────────────────────────────
# Lung - JSRT + Montgomery + NIH (Kaggle: nikhilpandey360/chest-xray-masks-and-labels)
# ──────────────────────────────────────────────────────────────────────

def download_lung(data_dir: Path, lung_zip: str = None):
    print("\n" + "=" * 56)
    print("  Lung Segmentation - Montgomery County CXR Set")
    print("=" * 56)

    lung_dir = data_dir / "lung"

    if lung_zip:
        zip_path = Path(lung_zip)
        if not zip_path.exists():
            print(f"  [lỗi] Không tìm thấy file: {lung_zip}")
            _print_lung_instructions()
            return
        _extract_lung(zip_path, lung_dir)
        return

    if lung_dir.exists() and len(list((lung_dir / "train" / "images").glob("*.png"))) > 0:
        n = len(list((lung_dir / "train" / "images").glob("*.png")))
        print(f"  ✓ Lung đã có {n} ảnh train tại {lung_dir}")
        return

    tmp = data_dir / "_lung_tmp"
    tmp.mkdir(exist_ok=True)

    # Tải trực tiếp từ NLM/NIH (public, không cần tài khoản, ~588 MB)
    _NLM_URL = "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip"
    zip_dst = tmp / "NLM-MontgomeryCXRSet.zip"
    print(f"  Tải Montgomery CXR Set từ NLM (~588 MB) ...")
    try:
        _download_with_progress(_NLM_URL, zip_dst)
    except Exception as e:
        print(f"\n  [lỗi khi tải] {e}")
        shutil.rmtree(tmp, ignore_errors=True)
        _print_lung_instructions()
        return

    _extract_lung(zip_dst, lung_dir)
    shutil.rmtree(tmp, ignore_errors=True)


def _organize_lung_from_flat(src_dir: Path, lung_dir: Path, test_ratio: float = 0.3):
    """
    Tổ chức ảnh từ flat directory => train/ và test/ theo cấu trúc project.
    Merge left/right lung masks thành một mask duy nhất nếu cần.
    Target: 407 train / 178 test (train_ratio ≈ 0.7).
    """
    import numpy as np
    from PIL import Image

    # Tìm tất cả ảnh X-quang và masks tương ứng
    img_exts = ('*.png', '*.jpg', '*.jpeg')
    all_imgs = []
    for ext in img_exts:
        all_imgs.extend(src_dir.rglob(ext))
    # Loại bỏ ảnh nằm trong thư mục masks
    all_imgs = [
        f for f in all_imgs
        if 'mask' not in f.parts[-2].lower() and 'mask' not in f.stem.lower()
    ]
    all_imgs = sorted(set(all_imgs))

    pairs = []
    for img_path in all_imgs:
        mask = _find_lung_mask(img_path, src_dir)
        if mask is not None:
            pairs.append((img_path, mask))

    if not pairs:
        print(f"\n  [!] Không tìm thấy cặp (ảnh, mask) trong {src_dir}")
        _print_lung_instructions()
        return

    # Shuffle và chia train/test
    import random
    rng = random.Random(42)
    rng.shuffle(pairs)
    n_test      = max(1, int(len(pairs) * test_ratio))
    test_pairs  = pairs[-n_test:]
    train_pairs = pairs[:-n_test]

    for split_name, split_pairs in [("train", train_pairs), ("test", test_pairs)]:
        img_out  = lung_dir / split_name / "images"
        mask_out = lung_dir / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_src, mask_src in split_pairs:
            out_name = img_src.stem + '.png'
            # Chuyển ảnh sang PNG nếu cần
            if not (img_out / out_name).exists():
                Image.open(img_src).convert('RGB').save(img_out / out_name)

            # Merge mask nếu là tuple (left, right lung masks)
            mask_dst = mask_out / out_name
            if not mask_dst.exists():
                if isinstance(mask_src, tuple):
                    left  = np.array(Image.open(mask_src[0]).convert('L'))
                    right = np.array(Image.open(mask_src[1]).convert('L'))
                    merged = np.maximum(left, right)
                    Image.fromarray(merged).save(mask_dst)
                else:
                    Image.open(mask_src).convert('L').save(mask_dst)

    _verify_lung(lung_dir)


def _find_lung_mask(img_path: Path, src_dir: Path):
    """
    Tìm mask tương ứng cho một ảnh X-quang.
    Trả về Path (single mask), tuple[Path, Path] (left+right masks), hoặc None.
    """
    stem = img_path.stem

    # 1. Single mask - thử các thư mục gần ảnh trước
    candidate_dirs = [
        img_path.parent.parent / 'masks',
        img_path.parent / 'masks',
        src_dir / 'masks',
    ]
    for mask_dir in candidate_dirs:
        if not mask_dir.exists():
            continue
        for mask_name in (f'{stem}.png', f'{stem}.jpg', f'{stem}_mask.png'):
            mp = mask_dir / mask_name
            if mp.exists():
                return mp

    # 2. Montgomery left+right - thử đường dẫn tương đối chuẩn
    mont_left  = img_path.parent.parent / 'ManualMask' / 'leftMask'  / f'{stem}.png'
    mont_right = img_path.parent.parent / 'ManualMask' / 'rightMask' / f'{stem}.png'
    if mont_left.exists() and mont_right.exists():
        return (mont_left, mont_right)

    # 3. Fallback: tìm toàn bộ src_dir bằng rglob (bất kể cấu trúc thư mục)
    left_hits  = list(src_dir.rglob(f'leftMask/{stem}.png'))
    right_hits = list(src_dir.rglob(f'rightMask/{stem}.png'))
    if left_hits and right_hits:
        return (left_hits[0], right_hits[0])

    # 4. Single mask rglob
    for mask_name in (f'{stem}_mask.png', f'{stem}.png'):
        hits = [
            p for p in src_dir.rglob(mask_name)
            if 'leftmask' not in str(p).lower()
            and 'rightmask' not in str(p).lower()
            and p != img_path
        ]
        if hits:
            return hits[0]

    return None


def _extract_lung(zip_path: Path, lung_dir: Path):
    print(f"  Đang giải nén {zip_path.name} ...", end="", flush=True)
    extract_dir = lung_dir.parent / "_lung_extract"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    print(" xong")
    _organize_lung_from_flat(extract_dir, lung_dir)
    shutil.rmtree(extract_dir, ignore_errors=True)


def _verify_lung(lung_dir: Path):
    for split in ('train', 'test'):
        img_dir = lung_dir / split / 'images'
        n_imgs  = sum(len(list(img_dir.glob(ext))) for ext in ('*.png', '*.jpg'))
        print(f"  ✓ lung/{split}: {n_imgs} ảnh  =>  {img_dir.parent}")
    print(f"    => {lung_dir}")


def _print_lung_instructions():
    print("""
  [!] Không thể tự động tải Lung dataset. Các cách tải thủ công:

  Cách 1 - Kaggle (Montgomery ~140 MB, không cần accept rules):
    1. Đặt kaggle.json vào ~/.kaggle/kaggle.json
    2. Chạy lại script này

  Cách 2 - Tải thủ công từ Kaggle web:
    1. Vào https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery
    2. Click "Download" => lưu zip về máy
    3. Chạy:
       python download_datasets.py --lung --lung_zip /path/to/montgomery.zip --data_dir data/

  Cách 3 - NIH public FTP (không cần tài khoản):
    https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-rays-Montgomery/
    Tải MontgomerySet.zip => giải nén => tổ chức thành:
      data/lung/train/images/*.png  + data/lung/train/masks/*.png
      data/lung/test/images/*.png   + data/lung/test/masks/*.png
""")


# ──────────────────────────────────────────────────────────────────────
# DSB-2018 - Data Science Bowl 2018 Nucleus Segmentation
# ──────────────────────────────────────────────────────────────────────

def download_dsb2018(data_dir: Path, dsb2018_zip: "str | None" = None):
    """
    Tải và xử lý Data Science Bowl 2018 nucleus segmentation.

    dsb2018_zip: path tới zip đã tải thủ công (data-science-bowl-2018.zip hoặc stage1_train.zip).
    Nếu không cung cấp, tự động tải qua Kaggle competitions API.

    Thứ tự ưu tiên (khi không có dsb2018_zip):
      1. Kaggle competitions (cần accept rules tại kaggle.com/c/data-science-bowl-2018/rules)
      2. Kaggle dataset mirror paultimothymooney/data-science-bowl-2018 (không cần accept)
    """
    print("\n" + "=" * 56)
    print("  Data Science Bowl 2018 - Nucleus Segmentation")
    print("=" * 56)

    dsb_dir  = data_dir / "dsb2018"
    img_dir  = dsb_dir / "images"
    msk_dir  = dsb_dir / "masks"

    if img_dir.exists() and len(list(img_dir.glob("*.png"))) >= 500:
        n = len(list(img_dir.glob("*.png")))
        print(f"  ✓ DSB-2018 đã có {n} ảnh tại {dsb_dir}")
        return

    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    # Dùng zip đã tải thủ công
    if dsb2018_zip:
        zip_path = Path(dsb2018_zip)
        if not zip_path.exists():
            print(f"  [lỗi] Không tìm thấy file: {dsb2018_zip}")
            _print_dsb2018_instructions()
            return
        print(f"  Đang giải nén {zip_path.name} ...", end="", flush=True)
        tmp_manual = data_dir / "_dsb2018_manual"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_manual)
        print(" xong")
        # Nếu bên trong có stage1_train.zip thì giải nén thêm một tầng
        inner_zips = list(tmp_manual.rglob("stage1_train.zip"))
        if inner_zips:
            print(f"  Đang giải nén stage1_train.zip ...", end="", flush=True)
            train_dir = tmp_manual / "_train"
            with zipfile.ZipFile(inner_zips[0]) as zf:
                zf.extractall(train_dir)
            print(" xong")
            _process_dsb2018(train_dir, img_dir, msk_dir)
        else:
            _process_dsb2018(tmp_manual, img_dir, msk_dir)
        shutil.rmtree(tmp_manual, ignore_errors=True)
        n_imgs  = len(list(img_dir.glob("*.png")))
        n_masks = len(list(msk_dir.glob("*.png")))
        print(f"\n  ✓ DSB-2018 sẵn sàng: {n_imgs} ảnh | {n_masks} masks  =>  {dsb_dir}")
        return

    if not shutil.which("kaggle"):
        _print_dsb2018_instructions()
        return

    tmp = data_dir / "_dsb2018_kaggle"
    tmp.mkdir(exist_ok=True)

    # Kaggle competitions - chỉ tải stage1_train.zip (có masks), bỏ qua stage2/test
    print("  Tải stage1_train.zip từ Kaggle competitions ...", end="", flush=True)
    ret = os.system(
        f'kaggle competitions download -c data-science-bowl-2018 '
        f'-f stage1_train.zip -p "{tmp}" -q 2>/dev/null'
    )
    if ret != 0:
        print(" thất bại (403 - cần accept rules)")
        shutil.rmtree(tmp, ignore_errors=True)
        _print_dsb2018_instructions()
        return
    print(" xong")

    stage1_zip = tmp / "stage1_train.zip"
    if not stage1_zip.exists():
        # Một số phiên bản Kaggle CLI đặt tên khác
        zips = list(tmp.glob("*.zip"))
        stage1_zip = zips[0] if zips else None

    if stage1_zip is None:
        print(f"  [lỗi] Không tìm thấy stage1_train.zip trong {tmp}")
        _print_dsb2018_instructions()
        return

    print(f"  Đang giải nén {stage1_zip.name} ...", end="", flush=True)
    train_dir = tmp / "_train"
    with zipfile.ZipFile(stage1_zip) as zf:
        zf.extractall(train_dir)
    print(" xong")

    _process_dsb2018(train_dir, img_dir, msk_dir)
    shutil.rmtree(tmp, ignore_errors=True)

    n_imgs  = len(list(img_dir.glob("*.png")))
    n_masks = len(list(msk_dir.glob("*.png")))
    print(f"\n  ✓ DSB-2018 sẵn sàng: {n_imgs} ảnh | {n_masks} masks")
    print(f"    => {dsb_dir}")


def _process_dsb2018(src_dir: Path, img_dir: Path, msk_dir: Path):
    """
    Flatten DSB-2018 raw competition structure:
      stage1_train/{image_id}/images/{image_id}.png
      stage1_train/{image_id}/masks/*.png  (multiple per image)
    => images/{image_id}.png
    => masks/{image_id}.png  (OR-merged binary mask)
    """
    import numpy as np
    from PIL import Image

    sample_dirs = [d for d in src_dir.iterdir() if d.is_dir()]
    if not sample_dirs:
        # Có thể đã có thêm một cấp thư mục
        sub = list(src_dir.iterdir())
        if sub and sub[0].is_dir():
            sample_dirs = [d for d in sub[0].iterdir() if d.is_dir()]

    print(f"  Đang xử lý {len(sample_dirs)} samples (merge nucleus masks) ...")
    processed = 0
    for sample_dir in sorted(sample_dirs):
        image_id = sample_dir.name

        # Tìm ảnh gốc
        img_candidates = list((sample_dir / 'images').glob('*.png'))
        if not img_candidates:
            img_candidates = list((sample_dir / 'images').glob('*.jpg'))
        if not img_candidates:
            continue
        img_src = img_candidates[0]

        # Merge tất cả nucleus masks bằng OR
        masks_dir = sample_dir / 'masks'
        mask_files = list(masks_dir.glob('*.png'))
        if not mask_files:
            continue

        ref = np.array(Image.open(img_src))
        H, W = ref.shape[:2]
        merged = np.zeros((H, W), dtype=np.uint8)
        for mf in mask_files:
            m = np.array(Image.open(mf).convert('L'))
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m).resize((W, H), Image.NEAREST))
            merged = np.maximum(merged, m)

        # Lưu ảnh và mask
        dst_img  = img_dir / f'{image_id}.png'
        dst_mask = msk_dir / f'{image_id}.png'
        if not dst_img.exists():
            Image.open(img_src).convert('RGB').save(dst_img)
        if not dst_mask.exists():
            Image.fromarray(merged).save(dst_mask)

        processed += 1
        if processed % 100 == 0:
            print(f"    {processed}/{len(sample_dirs)} done ...")

    print(f"  Đã xử lý {processed} samples.")


def _print_dsb2018_instructions():
    print("""
  [!] Không thể tự động tải DSB-2018. Cần accept competition rules trước.

  Cách 1 - Kaggle CLI (sau khi accept rules):
    1. Vào https://www.kaggle.com/competitions/data-science-bowl-2018/rules
       => click "I Understand and Accept"
    2. Chạy lại:
       python download_datasets.py --dsb2018 --data_dir data/

  Cách 2 - Tải thủ công stage1_train.zip:
    1. Vào https://www.kaggle.com/competitions/data-science-bowl-2018/data
    2. Download file "stage1_train.zip" (chứa images + masks, ~385 MB)
    3. Chạy:
       python download_datasets.py --dsb2018_zip /path/to/stage1_train.zip --data_dir data/
""")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _download_with_progress(url: str, dst: Path):
    """Tải file với progress bar đơn giản."""
    def _hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(count * block_size / total_size * 100, 100)
        bar = int(pct / 5)
        print(f"\r  {'█'*bar}{'░'*(20-bar)} {pct:5.1f}%", end="", flush=True)

    urlretrieve(url, dst, reporthook=_hook)
    print()


def _verify_structure(data_dir: Path):
    print("\n" + "=" * 56)
    print("  Kiểm tra cấu trúc cuối cùng")
    print("=" * 56)
    datasets = {
        "isic":    (data_dir / "isic"    / "images",            "*.jpg",  2500),
        "glas":    (data_dir / "glas"    / "train" / "images",  "*.bmp",    80),
        "covid":   (data_dir / "covid"   / "images",            "*.png",    50),
        "lung":    (data_dir / "lung"    / "train" / "images",  "*.png",   300),
        "dsb2018": (data_dir / "dsb2018" / "images",            "*.png",   500),
    }
    all_ok = True
    for name, (img_dir, pattern, min_count) in datasets.items():
        if img_dir.exists():
            n = len(list(img_dir.glob(pattern)))
            status = "✓" if n >= min_count else "⚠"
            print(f"  {status} {name:<8} {n:>5} ảnh  =>  {img_dir.parent}")
            if n < min_count:
                all_ok = False
        else:
            print(f"  ✗ {name:<8} chưa có  =>  {img_dir.parent}")
            all_ok = False
    print()
    if all_ok:
        print("  Tất cả datasets sẵn sàng. Ví dụ chạy training:")
        print(f"    python train.py --dataset isic    --data_root {data_dir}/isic    --model proposed")
        print(f"    python train.py --dataset lung    --data_root {data_dir}/lung    --model proposed")
        print(f"    python train.py --dataset dsb2018 --data_root {data_dir}/dsb2018 --model proposed")
    else:
        print("  Một số dataset chưa đầy đủ. Xem hướng dẫn ở trên.")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tải và chuẩn bị datasets cho thesis experiments"
    )
    p.add_argument("--all",       action="store_true", help="Tải tất cả datasets")
    p.add_argument("--isic",      action="store_true", help="Chỉ tải ISIC-2018")
    p.add_argument("--glas",      action="store_true", help="Chỉ tải GlaS")
    p.add_argument("--covid",     action="store_true", help="Tải COVID CC-CCII tự động")
    p.add_argument("--lung",      action="store_true", help="Tải Lung (JSRT+Montgomery) qua Kaggle")
    p.add_argument("--dsb2018",   action="store_true", help="Tải Data Science Bowl 2018 qua Kaggle")
    p.add_argument("--data_dir",  type=str, default="data", help="Thư mục lưu dữ liệu")
    p.add_argument("--glas_zip",  type=str, default=None,
                   help="Đường dẫn tới file zip GlaS đã tải thủ công")
    p.add_argument("--lung_zip",    type=str, default=None,
                   help="Đường dẫn tới file zip Lung dataset đã tải thủ công")
    p.add_argument("--dsb2018_zip", type=str, default=None,
                   help="Đường dẫn tới data-science-bowl-2018.zip hoặc stage1_train.zip đã tải thủ công")
    p.add_argument("--covid_zip", type=str, nargs="+", default=None,
                   help="Dùng zip đã tải thủ công thay vì tự tải (COVID19-*.zip ...)")
    p.add_argument("--covid_parts", type=int, nargs="+", default=None,
                   help="Chỉ tải một số zip cụ thể, vd: --covid_parts 1 2 3  (mặc định: 1-31)")
    p.add_argument("--lesions_csv", type=str, default=None,
                   help="Path tới lesions_slices.csv khi dùng --covid_zip thủ công")
    return p.parse_args()


def main():
    args     = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    run_isic    = args.all or args.isic
    run_glas    = args.all or args.glas    or args.glas_zip
    run_covid   = args.all or args.covid   or args.covid_zip or args.covid_parts
    run_lung    = args.all or args.lung    or args.lung_zip
    run_dsb2018 = args.all or args.dsb2018 or args.dsb2018_zip

    if not any([run_isic, run_glas, run_covid, run_lung, run_dsb2018]):
        print("Chưa chọn dataset. Dùng --all hoặc --isic / --glas / --covid / --lung / --dsb2018")
        print("Xem thêm: python download_datasets.py --help")
        sys.exit(0)

    if run_isic:
        download_isic(data_dir)

    if run_glas:
        download_glas(data_dir, glas_zip=args.glas_zip)

    if run_covid:
        if args.covid_zip:
            prepare_covid_from_zips(args.covid_zip, data_dir, lesions_csv=args.lesions_csv)
        else:
            download_covid(data_dir, parts=args.covid_parts)

    if run_lung:
        download_lung(data_dir, lung_zip=args.lung_zip)

    if run_dsb2018:
        download_dsb2018(data_dir, dsb2018_zip=args.dsb2018_zip)

    _verify_structure(data_dir)


if __name__ == "__main__":
    main()

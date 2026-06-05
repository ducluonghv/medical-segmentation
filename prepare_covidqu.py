"""
Extract COVID-QU-Ex infection segmentation data into flat images/ + masks/ structure.

Usage:
    python prepare_covidqu.py --zip /path/to/covidqu.zip --out data/covid

Output:
    data/covid/
        images/   covid_<id>_train.png | covid_<id>_val.png | covid_<id>_test.png
        masks/    (same filenames as images)

The _train/_val/_test suffix lets datasets/covid.py detect pre-defined splits
without downloading the full dataset multiple times.
"""
import argparse
import zipfile
from pathlib import Path


PREFIX = "Infection Segmentation Data/Infection Segmentation Data/"

SPLIT_PATHS = {
    "train": (f"{PREFIX}Train/COVID-19/images/",
              f"{PREFIX}Train/COVID-19/infection masks/"),
    "val":   (f"{PREFIX}Val/COVID-19/images/",
              f"{PREFIX}Val/COVID-19/infection masks/"),
    "test":  (f"{PREFIX}Test/COVID-19/images/",
              f"{PREFIX}Test/COVID-19/infection masks/"),
}


def extract(zip_path: Path, out_dir: Path):
    img_dir  = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()

        for split, (img_prefix, mask_prefix) in SPLIT_PATHS.items():
            imgs  = [m for m in members if m.startswith(img_prefix)  and m.endswith(".png")]
            masks = [m for m in members if m.startswith(mask_prefix) and m.endswith(".png")]

            # Build lookup: stem -> zip path
            mask_by_stem = {Path(m).name: m for m in masks}

            print(f"  {split}: {len(imgs)} images, {len(masks)} masks")
            extracted = 0
            for img_zip_path in imgs:
                stem     = Path(img_zip_path).stem          # e.g. "covid_1"
                img_name = f"{stem}_{split}.png"
                dst_img  = img_dir  / img_name
                dst_mask = mask_dir / img_name

                if dst_img.exists() and dst_mask.exists():
                    extracted += 1
                    continue

                orig_name = Path(img_zip_path).name          # "covid_1.png"
                mask_zip_path = mask_by_stem.get(orig_name)
                if mask_zip_path is None:
                    continue

                with zf.open(img_zip_path)  as f: dst_img.write_bytes(f.read())
                with zf.open(mask_zip_path) as f: dst_mask.write_bytes(f.read())
                extracted += 1

            print(f"    => {extracted} pairs saved")

    n_imgs  = len(list(img_dir.glob("*.png")))
    n_masks = len(list(mask_dir.glob("*.png")))
    print(f"\nDone: {n_imgs} images | {n_masks} masks  =>  {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Prepare COVID-QU-Ex for thesis experiments")
    p.add_argument("--zip", required=True,  help="Path to covidqu.zip downloaded from Kaggle")
    p.add_argument("--out", default="data/covid", help="Output directory (default: data/covid)")
    args = p.parse_args()

    extract(Path(args.zip), Path(args.out))


if __name__ == "__main__":
    main()

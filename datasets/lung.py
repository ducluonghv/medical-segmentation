"""
Lung Segmentation Dataset - Montgomery County Chest X-ray Set (NLM/NIH)

Expected folder layout (sau khi chạy download_datasets.py --lung):
    <root>/
        train/
            images/   *.png  (~96 chest X-ray images, grayscale)
            masks/    *.png  (binary lung masks - left+right merged)
        test/
            images/   *.png  (~42 images)
            masks/    *.png

Dataset: Montgomery County CXR Set - 138 ảnh PA chest X-ray
         (bình thường và bệnh nhân lao phổi)
Split: ~96 train / ~10 val / ~32 test  (val_ratio=0.1 from train, test_ratio=0.3)
Input resize: 256×256

Download (Kaggle, public, ~140 MB, không cần accept rules):
  kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery
  Sau đó: python download_datasets.py --lung --data_dir data/

Raw Montgomery structure (bên trong zip):
  MontgomerySet/
    CXR_png/          *.png  (138 images)
    ManualMask/
      leftMask/       *.png
      rightMask/      *.png
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class LungDataset(Dataset):
    """Chest X-ray lung segmentation (JSRT + Montgomery + NIH combined).

    Args:
        root      : dataset root (must contain train/ and test/ sub-dirs)
        split     : 'train' | 'val' | 'test'
        img_size  : resize resolution (default 256)
        augment   : random augmentation on train split
        val_ratio : fraction of train images used for validation
        seed      : reproducibility seed
    """

    # CXR images are grayscale; normalise with neutral stats
    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    def __init__(
        self,
        root:      str,
        split:     str   = 'train',
        img_size:  int   = 256,
        augment:   bool  = True,
        val_ratio: float = 0.1,
        seed:      int   = 42,
    ):
        assert split in ('train', 'val', 'test')
        self.split    = split
        self.img_size = img_size
        self.augment  = augment and split == 'train'

        root = Path(root)

        if split in ('train', 'val'):
            pairs = self._collect_pairs(root / 'train' / 'images', root / 'train' / 'masks')
            rng = random.Random(seed)
            rng.shuffle(pairs)
            n_val       = max(1, int(len(pairs) * val_ratio))
            train_pairs = pairs[:-n_val]
            val_pairs   = pairs[-n_val:]
            self.pairs  = train_pairs if split == 'train' else val_pairs
        else:
            self.pairs = self._collect_pairs(root / 'test' / 'images', root / 'test' / 'masks')

        if not self.pairs:
            raise FileNotFoundError(f"No (image, mask) pairs found under {root}/{split}")

        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)

    @staticmethod
    def _collect_pairs(img_dir: Path, mask_dir: Path):
        pairs = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            for img_path in sorted(img_dir.glob(ext)):
                stem = img_path.stem
                # Support common mask naming conventions
                for mask_name in (
                    f'{stem}.png',
                    f'{stem}_mask.png',
                    f'{stem}.jpg',
                ):
                    mp = mask_dir / mask_name
                    if mp.exists():
                        pairs.append((str(img_path), str(mp)))
                        break
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert('RGB')   # grayscale CXR => 3ch
        mask = Image.open(mask_path).convert('L')

        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t  = self.normalize(TF.to_tensor(img))
        mask_t = torch.from_numpy(
            (np.array(mask) > 127).astype(np.float32)
        ).unsqueeze(0)

        return img_t, mask_t, img_path

    def _augment(self, img, mask):
        # Horizontal flip only - vertical flip is anatomically wrong for CXR
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        # Small rotation: lungs are roughly symmetric, allow slight tilt
        angle = random.uniform(-10, 10)
        img   = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

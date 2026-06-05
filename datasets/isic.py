"""
ISIC-2018 Skin Lesion Segmentation Dataset

Expected folder layout:
    <root>/
        images/   *.jpg   (2596 dermoscopy images)
        masks/    *.png   (binary segmentation masks, pixel value 0 or 255)

Split: ISIC-2018 (2596 ảnh tổng)
       1870 train / 207 val / 519 test  (72% / 8% / 20%)
Input resize: 256×256

Download: https://challenge.isic-archive.com/data/#2018
"""
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class ISICDataset(Dataset):
    """ISIC-2018 Task 1 skin lesion segmentation.

    Args:
        root      : dataset root directory (must contain images/ and masks/)
        split     : 'train' | 'val' | 'test'
        img_size  : spatial resolution to resize to (default 256)
        augment   : apply random augmentation (train split only)
        val_ratio : fraction of training images held out for validation
        seed      : random seed for train/val split
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root:      str,
        split:     str  = 'train',
        img_size:  int  = 256,
        augment:   bool = True,
        val_ratio: float = 0.1,
        seed:      int  = 42,
    ):
        assert split in ('train', 'val', 'test')
        self.split    = split
        self.img_size = img_size
        self.augment  = augment and split == 'train'

        img_dir  = Path(root) / 'images'
        mask_dir = Path(root) / 'masks'

        all_imgs = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        if not all_imgs:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # Derive mask path from image name
        pairs = []
        for img_path in all_imgs:
            stem = img_path.stem
            # ISIC masks are named <stem>_segmentation.png or <stem>.png
            for suffix in (f'{stem}_segmentation.png', f'{stem}.png'):
                mp = mask_dir / suffix
                if mp.exists():
                    pairs.append((str(img_path), str(mp)))
                    break

        if not pairs:
            raise FileNotFoundError(f"No (image, mask) pairs found under {root}")

        # Reproducible train/val/test split
        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_test  = int(len(pairs) * 0.2)           # 20 % test (≈ 520 for 2596 total)
        n_val   = int((len(pairs) - n_test) * val_ratio)
        test_pairs  = pairs[-n_test:]
        train_pairs = pairs[:-n_test]
        val_pairs   = train_pairs[-n_val:]
        train_pairs = train_pairs[:-n_val]

        self.pairs = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}[split]

        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img, mask = self._resize(img, mask)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t  = TF.to_tensor(img)            # (3, H, W) float [0,1]
        img_t  = self.normalize(img_t)
        mask_t = torch.from_numpy(
            (np.array(mask) > 127).astype(np.float32)
        ).unsqueeze(0)                         # (1, H, W) binary

        return img_t, mask_t, img_path

    # ------------------------------------------------------------------
    def _resize(self, img, mask):
        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        return img, mask

    def _augment(self, img, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)
        # Random vertical flip
        if random.random() > 0.5:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)
        # Random rotation ±30°
        angle = random.uniform(-30, 30)
        img   = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        # Color jitter (image only)
        img = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)(img)
        return img, mask

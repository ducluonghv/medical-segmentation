"""
GlaS (Gland Segmentation) Dataset - MICCAI 2015 Challenge

Expected folder layout:
    <root>/
        train/
            images/   train_1.bmp, train_2.bmp, ...
            masks/    train_1_anno.bmp, train_2_anno.bmp, ...
        test/
            images/   testA_1.bmp, ...  (or testB_*)
            masks/    testA_1_anno.bmp, ...

Split: GlaS (165 ảnh tổng)
       77 train / 8 val / 80 test  (47% / 5% / 48%)
Input resize: 128×128

Download: https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/
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


class GlaSDataset(Dataset):
    """GlaS gland segmentation.

    Args:
        root     : dataset root (must contain train/ and test/ sub-dirs)
        split    : 'train' | 'val' | 'test'
        img_size : resize resolution (default 128, matching paper)
        augment  : random augmentation on train split
        val_ratio: fraction of train images used for validation
        seed     : reproducibility seed
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root:      str,
        split:     str  = 'train',
        img_size:  int  = 128,
        augment:   bool = True,
        val_ratio: float = 0.1,
        seed:      int  = 42,
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
            raise FileNotFoundError(f"No (image, mask) pairs found in {root}/{split}")

        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)

    # ------------------------------------------------------------------
    @staticmethod
    def _collect_pairs(img_dir: Path, mask_dir: Path):
        pairs = []
        for ext in ('*.bmp', '*.png', '*.jpg'):
            for img_path in sorted(img_dir.glob(ext)):
                stem = img_path.stem
                # GlaS naming: train_1.bmp => mask train_1_anno.bmp
                for mask_name in (f'{stem}_anno{img_path.suffix}', f'{stem}.png'):
                    mp = mask_dir / mask_name
                    if mp.exists():
                        pairs.append((str(img_path), str(mp)))
                        break
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t  = T.Normalize(mean=self.MEAN, std=self.STD)(TF.to_tensor(img))
        mask_t = torch.from_numpy(
            (np.array(mask) > 0).astype(np.float32)
        ).unsqueeze(0)

        return img_t, mask_t, img_path

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)
        angle = random.uniform(-30, 30)
        img   = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

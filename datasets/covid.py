"""
COVID-19 Infection Segmentation — COVID-QU-Ex Dataset

Source: Kaggle anasmohammedtahir/covidqu
        "Infection Segmentation Data" subset — COVID-19 class only

Expected folder layout (flat, after prepare_covidqu.py):
    <root>/
        images/   *.png  (2913 chest X-ray slices)
        masks/    *.png  (binary infection masks, same filenames)

Split (pre-defined in COVID-QU-Ex, preserved via stem suffix):
    Train: 1864 | Val: 466 | Test: 583
    Detected automatically from filename: covid_<id>_train / _val / _test
    Fallback: random 64%/16%/20% split if suffix not present.

Input resize: 256×256
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class COVIDDataset(Dataset):
    """COVID-19 infection segmentation on chest X-ray images (COVID-QU-Ex).

    Args:
        root      : dataset root (must contain images/ and masks/)
        split     : 'train' | 'val' | 'test'
        img_size  : resize resolution (default 256)
        augment   : random augmentation on train split
        seed      : reproducibility seed (used only for fallback random split)
    """

    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    def __init__(
        self,
        root:     str,
        split:    str  = 'train',
        img_size: int  = 256,
        augment:  bool = True,
        seed:     int  = 42,
    ):
        assert split in ('train', 'val', 'test')
        self.split    = split
        self.img_size = img_size
        self.augment  = augment and split == 'train'

        img_dir  = Path(root) / 'images'
        mask_dir = Path(root) / 'masks'

        all_imgs = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
        if not all_imgs:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # Build (image, mask) pairs
        pairs = []
        for img_path in all_imgs:
            mp = mask_dir / img_path.name
            if not mp.exists():
                mp = mask_dir / (img_path.stem + '.png')
            if mp.exists():
                pairs.append((str(img_path), str(mp)))

        if not pairs:
            raise FileNotFoundError(f"No (image, mask) pairs found in {root}")

        # Try pre-split: filenames end with _train / _val / _test before extension
        train_pairs = [p for p in pairs if Path(p[0]).stem.endswith('_train')]
        val_pairs   = [p for p in pairs if Path(p[0]).stem.endswith('_val')]
        test_pairs  = [p for p in pairs if Path(p[0]).stem.endswith('_test')]

        if train_pairs or val_pairs or test_pairs:
            # Pre-split filenames present
            splits = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}
        else:
            # Fallback: random split  64% / 16% / 20%
            rng = random.Random(seed)
            rng.shuffle(pairs)
            n      = len(pairs)
            n_test = max(1, int(n * 0.20))
            n_val  = max(1, int(n * 0.16))
            test_pairs  = pairs[-n_test:]
            rest        = pairs[:-n_test]
            val_pairs   = rest[-n_val:]
            train_pairs = rest[:-n_val]
            splits = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}

        self.pairs     = splits[split]
        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)

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

        img_t  = self.normalize(TF.to_tensor(img))
        mask_t = torch.from_numpy(
            (np.array(mask) > 127).astype(np.float32)
        ).unsqueeze(0)

        return img_t, mask_t, img_path

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)
        angle = random.uniform(-15, 15)
        img   = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

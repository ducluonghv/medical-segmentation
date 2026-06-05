"""
Clean-CC-CCII COVID-19 Pneumonia Lesion Segmentation Dataset

Expected folder layout:
    <root>/
        images/   *.png  (260 chest CT slices, pixel range [0,255])
        masks/    *.png  (binary infection masks)

Split: COVID (260 ảnh tổng)
       180 train / 20 val / 60 test  (69% / 8% / 23%)
Input resize: 512×512

Download: http://ncov-ai.big.ac.cn/download
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
    """COVID-19 pneumonia lesion segmentation on chest CT slices.

    Args:
        root      : dataset root (must contain images/ and masks/)
        split     : 'train' | 'val' | 'test'
        img_size  : resize resolution (default 512, matching paper)
        augment   : random augmentation on train split
        val_ratio : fraction of train set for validation
        seed      : reproducibility seed
    """

    # CT images are grayscale; replicate to 3 channels for consistency
    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    def __init__(
        self,
        root:      str,
        split:     str  = 'train',
        img_size:  int  = 512,
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

        all_imgs = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))
        if not all_imgs:
            raise FileNotFoundError(f"No images found in {img_dir}")

        pairs = []
        for img_path in all_imgs:
            mp = mask_dir / img_path.name
            if not mp.exists():
                mp = mask_dir / (img_path.stem + '.png')
            if mp.exists():
                pairs.append((str(img_path), str(mp)))

        if not pairs:
            raise FileNotFoundError(f"No (image, mask) pairs found in {root}")

        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_test      = 60
        n_val       = max(1, int((len(pairs) - n_test) * val_ratio))
        test_pairs  = pairs[-n_test:]
        train_pairs = pairs[:-n_test]
        val_pairs   = train_pairs[-n_val:]
        train_pairs = train_pairs[:-n_val]

        self.pairs = {'train': train_pairs, 'val': val_pairs, 'test': test_pairs}[split]
        self.normalize = T.Normalize(mean=self.MEAN, std=self.STD)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img  = Image.open(img_path).convert('RGB')    # grayscale => 3ch
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
        angle = random.uniform(-15, 15)    # smaller range for CT
        img   = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return img, mask

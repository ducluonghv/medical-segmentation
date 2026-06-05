"""
Data Science Bowl 2018 - Nucleus Segmentation

Expected folder layout (sau khi chạy download_datasets.py --dsb2018):
    <root>/
        images/   *.png  (671 ảnh kính hiển vi đa phương thức, đã flatten)
        masks/    *.png  (binary mask - các nhân tế bào đã merge thành 1 mask)

Split: DSB-2018 (671 ảnh từ 15 thí nghiệm)
       537 train / 67 val / 67 test  (fixed counts theo đề cương)
Input resize: 256×256

Raw competition structure (trước khi xử lý):
    stage1_train/
        {image_id}/
            images/   {image_id}.png
            masks/    *.png  (mỗi file = 1 nhân riêng lẻ => cần OR lại)

Download (Kaggle):
  kaggle competitions download -c data-science-bowl-2018
  Sau đó: python download_datasets.py --dsb2018 --data_dir data/
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DSB2018Dataset(Dataset):
    """Data Science Bowl 2018 nucleus segmentation.

    Args:
        root      : dataset root (must contain images/ and masks/)
        split     : 'train' | 'val' | 'test'
        img_size  : resize resolution (default 256)
        augment   : random augmentation on train split
        seed      : reproducibility seed
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    # Fixed split sizes matching proposal (537 / 67 / 67)
    N_TEST = 67
    N_VAL  = 67

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

        pairs = []
        for img_path in all_imgs:
            mp = mask_dir / (img_path.stem + '.png')
            if mp.exists():
                pairs.append((str(img_path), str(mp)))

        if not pairs:
            raise FileNotFoundError(f"No (image, mask) pairs found under {root}")

        rng = random.Random(seed)
        rng.shuffle(pairs)

        n_test      = self.N_TEST
        n_val       = self.N_VAL
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
        angle = random.uniform(-30, 30)
        img   = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask  = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        # Colour jitter: DSB2018 spans nhiều modality nên augment mạnh hơn
        img = T.ColorJitter(brightness=0.3, contrast=0.3)(img)
        return img, mask

from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, return_weight=False):
        """
        image_dir: .../train/images
        mask_dir : .../train/masks
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.return_weight = return_weight

        # 이미지 파일 목록 정렬
        self.image_paths = sorted(list(self.image_dir.glob("*")))
        assert len(self.image_paths) > 0, "이미지 파일이 없습니다."

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        # 흑백 이미지 → 1채널
        img = Image.open(path).convert("L")   # (H, W)
        img = np.array(img, dtype=np.float32)

        # 0~255 → 0~1 정규화
        img = img / 255.0

        # (H, W) → (1, H, W)
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img)  # float32 tensor

    def _load_mask(self, path):
        # 마스크도 흑백으로 읽고 0/1로 이진화
        mask = Image.open(path).convert("L")
        mask = np.array(mask, dtype=np.uint8)

        # 0/255 → 0/1
        mask = (mask > 0).astype(np.uint8)

        # (H, W) 유지 (channel 필요 없음)
        return torch.from_numpy(mask)  # uint8 tensor

    def _make_weight_map(self, mask):
        """
        경계 강조용 weight map.
        일단 기본 버전: 모든 픽셀 1.0
        → 원하면 이후 논문처럼 경계 weight 강조 가능
        """
        weight = torch.ones_like(mask, dtype=torch.float32)
        return weight

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name  # 같은 파일명 기준

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # transform 쓰고 싶으면 여기서
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.return_weight:
            weight_map = self._make_weight_map(mask)
            return image, mask, weight_map
        else:
            return image, mask

def get_loaders(
    root_dir,
    batch_size=1,
    num_workers=4,
    return_weight=False,
):
    root = Path(root_dir)

    train_dataset = SegmentationDataset(
        image_dir=root / "data" / "train" / "images",
        mask_dir=root / "data" / "train" / "masks",
        return_weight=return_weight,
    )

    val_dataset = SegmentationDataset(
        image_dir=root / "data" / "val" / "images",
        mask_dir=root / "data" / "val" / "masks",
        return_weight=return_weight,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

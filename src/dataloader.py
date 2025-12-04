from pathlib import Path

import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

import torch
from torch.utils.data import Dataset, DataLoader

def elastic_deformation(image, mask, alpha=20.0, sigma=3.0, random_state=None):
    """
    U-Net 논문에서 말한 elastic deformation을 단순화해서 구현한 버전.
    - image: (H, W) float32, 0~1
    - mask : (H, W) uint8, 0/1

    alpha: 변형 강도 (크면 많이 휘어짐)
    sigma: 변형을 얼마나 부드럽게 할지 (가우시안 블러 크기)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape  # (H, W)

    # -1 ~ 1 사이 랜덤 필드 생성
    dx = random_state.uniform(-1, 1, size=shape)
    dy = random_state.uniform(-1, 1, size=shape)

    # 가우시안 필터로 부드럽게 만들고 alpha 배
    dx = ndimage.gaussian_filter(dx, sigma=sigma, mode="reflect") * alpha
    dy = ndimage.gaussian_filter(dy, sigma=sigma, mode="reflect") * alpha

    # 좌표 그리드 생성
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # 변형된 좌표
    indices = (y + dy, x + dx)

    # 이미지: bilinear(order=1), 마스크: nearest(order=0)
    image_def = ndimage.map_coordinates(
        image, indices, order=1, mode="reflect"
    )
    mask_def = ndimage.map_coordinates(
        mask,  indices, order=0, mode="nearest"
    )

    return image_def, mask_def

class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,
        return_weight=False,
        # elastic
        use_elastic=False,
        elastic_alpha=20.0,
        elastic_sigma=3.0,
        elastic_prob=0.5,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.return_weight = return_weight
        
        # elastic 옵션 추가
        self.use_elastic = use_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob

        self.image_paths = sorted(list(self.image_dir.glob("*")))
        if not self.image_paths:
            raise ValueError("이미지 파일이 없습니다.")

        missing_masks = []
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                missing_masks.append(mask_path)
            else:
                self.mask_paths.append(mask_path)

        if missing_masks:
            missing_names = ", ".join(p.name for p in missing_masks)
            raise FileNotFoundError(
                f"마스크 파일이 누락되었습니다: {missing_names}"
            )

        if len(self.mask_paths) != len(self.image_paths):
            raise ValueError("이미지와 마스크의 개수가 일치하지 않습니다.")

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        img = Image.open(path).convert("L")   # (H, W)
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        return torch.from_numpy(img)       # float32 tensor

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        return torch.from_numpy(mask)      # uint8 tensor

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
        mask_path = self.mask_paths[idx]

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        # elastic deformation
        if self.use_elastic and np.random.rand() < self.elastic_prob:
            # torch -> numpy (채널 빼고)
            img_np = image.squeeze(0).numpy()   # (H, W)
            mask_np = mask.numpy()             # (H, W)

            img_np, mask_np = elastic_deformation(
                img_np,
                mask_np,
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma,
            )

            # 다시 torch로 변환
            image = torch.from_numpy(img_np).unsqueeze(0).float()
            mask = torch.from_numpy(mask_np).to(mask.dtype)

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
        use_elastic=True,        
        elastic_alpha=20.0,
        elastic_sigma=3.0,
        elastic_prob=0.7,          # 70% 확률로 적용
    )

    val_dataset = SegmentationDataset(
        image_dir=root / "data" / "val" / "images",
        mask_dir=root / "data" / "val" / "masks",
        return_weight=return_weight,
        use_elastic=False,         # validation에는 augmentation X
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

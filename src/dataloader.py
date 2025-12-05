from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage # elastic deformation
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

def elastic_deformation(image, mask, alpha=10.0, sigma=4.0, random_state=None):
    """
    논문에서 사용된 elastic deformation 구현
    alpha : 변형 강도
    sigma : Gaussian smoothing 정도
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape  
    
    # -1 ~ 1 사이 초기 displacement field
    dx = random_state.uniform(-1, 1, size=shape)
    dy = random_state.uniform(-1, 1, size=shape)
    
    # displacement를 Gaussian smoothing하여 연속성 확보
    dx = ndimage.gaussian_filter(dx, sigma=sigma, mode="reflect") * alpha
    dy = ndimage.gaussian_filter(dy, sigma=sigma, mode="reflect") * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy, x + dx)
    
    # 이미지는 bilinear interpolation(order=1)
    # 마스크는 nearest interpolation(order=0)로 왜곡
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
        return_weight=False,
        # elastic
        use_elastic=False,
        elastic_alpha=10.0,
        elastic_sigma=4.0,
        elastic_prob=0.3,   # elastic 적용 확률
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.return_weight = return_weight
        
        self.use_elastic = use_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_prob = elastic_prob

        self.image_paths = sorted(list(self.image_dir.glob("*")))
        if not self.image_paths:
            raise ValueError("이미지 파일이 없습니다.")

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, path):
        """
        흑백 이미지를 0~1 float tensor (1, H, W)로 변환
        U-Net 입력 형식에 맞게 채널 차원을 앞에 추가
        """
        img = Image.open(path).convert("L")
        img = np.array(img, dtype=np.float32)    
        img = img / 255.0   
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img)  # float32 tensor, shape (1, H, W)

    def _load_mask(self, path):
        """
        마스크를 흑백으로 읽어서 0,1 이진 텐서로 변환
        """
        mask = Image.open(path).convert("L")
        mask = np.array(mask, dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)  # 0/255 → 0/1
        return torch.from_numpy(mask)  # uint8 tensor, shape (H, W)

    def _make_weight_map(self, mask):
        """
        논문 U-Net 방식에서 distance transform을 간단히 적용
        경계에 가까울수록 weight가 높아짐
        """
        mask_np = mask.cpu().numpy().astype(np.uint8)

        dist_fg = cv2.distanceTransform(mask_np, distanceType=cv2.DIST_L2, maskSize=5)
        dist_bg = cv2.distanceTransform(1 - mask_np, cv2.DIST_L2, 5)

        # 경계는 fg와 bg의 두 distance 중 최소가 되는 영역과 일치
        boundary_distance = np.minimum(dist_fg, dist_bg)

        # boundary 근처는 값이 낮음 → weight는 exp로 반비례 증가
        sigma = 5.0
        w0 = 5.0

        weight = 1.0 + w0 * np.exp(-(boundary_distance ** 2) / (2 * sigma * sigma))

        return torch.from_numpy(weight.astype(np.float32))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / self.image_paths[idx].name

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        # elastic deformation
        if self.use_elastic and np.random.rand() < self.elastic_prob:
            # torch -> numpy (채널 빼고)
            img_np = image.squeeze(0).numpy()   # (H, W)
            mask_np = mask.numpy()

            img_np, mask_np = elastic_deformation(
                img_np,
                mask_np,
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma,
            )
            # 다시 torch로 변환
            image = torch.from_numpy(img_np).unsqueeze(0).float()
            mask = torch.from_numpy(mask_np).to(mask.dtype)

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
    """
    프로젝트 루트(root_dir) 기준으로
    data/train, data/val 아래에서 train/val DataLoader 생성.
    """
    root = Path(root_dir)

    train_dataset = SegmentationDataset(
        image_dir=root / "data" / "train" / "images",
        mask_dir=root / "data" / "train" / "masks",
        return_weight=return_weight,
        use_elastic=True,        
        elastic_alpha=10.0,
        elastic_sigma=4.0,
        elastic_prob=0.3,       
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

"""
U-Net: Convolutional Networks for Biomedical Image Segmentation 논문 구조를
기본으로 한 학습 스크립트.

핵심 반영 내용:
- batch size = 1 (큰 타일 기준)
- padding 없는 conv 사용 시, 출력 크기에 맞춘 중앙 crop
- pixel-wise softmax + cross entropy
- class imbalance / 경계 강조를 위한 weight map(선택)
"""

from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_loaders   # SegmentationDataset 포함
from model import UNet               # PyTorch nn.Module 버전이라고 가정


def get_device() -> torch.device:
    """GPU가 있으면 cuda, 아니면 cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def center_crop_to(
    tensor: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    U-Net 원 논문처럼 padding=0인 conv를 쓰면 출력 feature map 크기가 줄어듦.
    이때 GT mask나 weight map을 출력 크기에 맞게 중앙 기준으로 crop.

    tensor: (N, H, W) 형태라고 가정
    target_hw: (H_out, W_out)
    """
    _, h, w = tensor.shape
    th, tw = target_hw

    if h == th and w == tw:
        return tensor

    y1 = (h - th) // 2
    x1 = (w - tw) // 2
    y2 = y1 + th
    x2 = x1 + tw

    return tensor[:, y1:y2, x1:x2]


def compute_weighted_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight_map: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    pixel-wise softmax + cross entropy + optional weight map.

    - logits: (N, C, H, W)
    - target: (N, H, W)  # 각 픽셀의 클래스 인덱스 (0 ~ C-1)
    - weight_map: (N, H, W)  # 경계 픽셀에 더 큰 weight 주고 싶을 때
    - class_weights: (C,)  # foreground / background 비율 보정용
    """
    # reduction='none'으로 픽셀별 loss 유지
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="none",
    )

    # CrossEntropyLoss는 내부에서 softmax까지 수행하므로 logits 그대로 사용
    per_pixel_loss = criterion(logits, target)  # (N, H, W)

    if weight_map is not None:
        # U-Net 논문처럼 경계 픽셀에 더 큰 loss를 주고 싶다면
        per_pixel_loss = per_pixel_loss * weight_map

    return per_pixel_loss.mean()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    """하나의 epoch에 대해 학습 진행."""
    model.train()
    running_loss = 0.0

    for batch in loader:
        # dataloader.py 구조:
        # return_weight=True면 (image, mask, weight_map)
        # 아니면 (image, mask)
        if len(batch) == 3:
            imgs, masks, weight_maps = batch
            weight_maps = weight_maps.to(device)
        else:
            imgs, masks = batch
            weight_maps = None

        imgs = imgs.to(device)          # (N, 1, H, W)
        masks = masks.to(device).long() # (N, H, W)

        optimizer.zero_grad()

        # forward
        logits = model(imgs)  # (N, C, H_out, W_out)

        # 출력 크기와 mask 크기가 다르면 중앙 기준 crop
        if logits.shape[2:] != masks.shape[1:]:
            masks = center_crop_to(masks, logits.shape[2:])
            if weight_maps is not None:
                weight_maps = center_crop_to(weight_maps, logits.shape[2:])

        loss = compute_weighted_ce_loss(
            logits,
            masks,
            weight_map=weight_maps,
            class_weights=class_weights,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    """validation set에 대한 평균 loss 계산."""
    model.eval()
    running_loss = 0.0

    for batch in loader:
        if len(batch) == 3:
            imgs, masks, weight_maps = batch
            weight_maps = weight_maps.to(device)
        else:
            imgs, masks = batch
            weight_maps = None

        imgs = imgs.to(device)
        masks = masks.to(device).long()

        logits = model(imgs)

        if logits.shape[2:] != masks.shape[1:]:
            masks = center_crop_to(masks, logits.shape[2:])
            if weight_maps is not None:
                weight_maps = center_crop_to(weight_maps, logits.shape[2:])

        loss = compute_weighted_ce_loss(
            logits,
            masks,
            weight_map=weight_maps,
            class_weights=class_weights,
        )
        running_loss += loss.item()

    return running_loss / len(loader)


def main():
    # ---------------- 경로 및 환경 설정 ----------------
    project_root = Path(__file__).resolve().parent.parent

    device = get_device()
    print(f"device: {device}")

    # reproducibility 어느 정도 맞추기
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    # ---------------- 데이터 로더 ----------------
    # 논문에서 batch size=1 사용 → 여기서도 기본값 1로.
    train_loader, val_loader = get_loaders(
        root_dir=project_root,
        batch_size=1,
        num_workers=0,      # Windows면 0 추천, Linux면 4 정도로 늘려도 됨
        return_weight=False # 경계 weight map 쓰려면 True로 변경
    )

    # ---------------- 모델 구성 ----------------
    # in_channels: 흑백이면 1, RGB면 3
    # num_classes: 배경/전경 = 2 (필요 시 변경)
    in_channels = 1
    num_classes = 2
    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    # ---------------- 최적화 / 손실 설정 ----------------
    # U-Net 논문: SGD + momentum 0.99 사용.
    # 여기서도 그 설정을 기본값으로 사용.
    lr = 1e-3
    momentum = 0.99
    weight_decay = 5e-4

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=False,
    )

    # class imbalance 보정: 배경/전경 비율 보고 조정 가능
    # 예시: 배경이 훨씬 많다면 foreground weight를 더 크게.
    class_weights = torch.tensor([0.5, 1.5], dtype=torch.float32).to(device)

    num_epochs = 50
    best_val_loss = float("inf")
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ---------------- 학습 루프 ----------------
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights=class_weights,
        )
        val_loss = evaluate(
            model,
            val_loader,
            device,
            class_weights=class_weights,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}"
        )

        # validation loss 기준으로 best model 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = ckpt_dir / "unet_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ best model updated: {ckpt_path}")

    print("Training finished.")


if __name__ == "__main__":
    main()

"""
U-Net: Convolutional Networks for Biomedical Image Segmentation 논문 구조를
기본으로 한 학습 스크립트.

핵심 반영 내용:
- batch size = 1 (큰 타일 기준)
- padding 없는 conv 사용 시, 출력 크기에 맞춘 중앙 crop (현재 모델은 padding=1이지만 형태 유지)
- pixel-wise softmax + cross entropy
- class imbalance / 경계 강조를 위한 weight map(선택)
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_loaders   # SegmentationDataset 포함
from model import UNet               # PyTorch nn.Module


def get_device() -> torch.device:
    """GPU가 있으면 cuda, 아니면 cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def center_crop_to(
    tensor: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    출력 feature map 크기에 맞게 GT mask나 weight map을 중앙 기준 crop.

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
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="none",
    )

    per_pixel_loss = criterion(logits, target)  # (N, H, W)

    if weight_map is not None:
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
        if len(batch) == 3:
            imgs, masks, weight_maps = batch
            weight_maps = weight_maps.to(device)
        else:
            imgs, masks = batch
            weight_maps = None

        imgs = imgs.to(device)          # (N, 1, H, W)
        masks = masks.to(device).long() # (N, H, W)

        optimizer.zero_grad()

        logits = model(imgs)            # (N, C, H_out, W_out)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net segmentation model")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="프로젝트 루트(데이터가 data/ 아래에 있다고 가정)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers (윈도우는 0 권장)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="훈련 epoch 수") # epoch 수 여기서 결정
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
    parser.add_argument(
        "--momentum", type=float, default=0.99, help="SGD momentum 값",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="L2 weight decay",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="체크포인트 저장 경로(기본: <root>/checkpoints)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- 경로 및 환경 설정 ----------------
    project_root = args.root
    ckpt_dir = args.checkpoint_dir or (project_root / "checkpoints")

    device = get_device()
    print(f"device: {device}")

    # reproducibility 어느 정도 맞추기
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # 연산 결정성을 위해 cudnn 설정과 별개로 명시
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ---------------- 데이터 로더 ----------------
    train_loader, val_loader = get_loaders(
        root_dir=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_weight=False,  # 경계 weight map 쓰려면 True로 변경
    )

    # ---------------- 모델 구성 ----------------
    in_channels = 1
    num_classes = 2
    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    # ---------------- 최적화 / 손실 설정 ----------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False,
    )

    # class imbalance 보정: 배경/전경 비율 보고 조정 가능
    class_weights = torch.tensor([0.5, 1.5], dtype=torch.float32).to(device)

    num_epochs = args.epochs
    best_val_loss = float("inf")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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

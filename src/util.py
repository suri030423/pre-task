# src/util.py
"""
U-Net 학습/평가 공용 유틸 함수 모음.

- logits_to_probs / logits_to_mask : 모델 출력 → 확률 / 바이너리 마스크
- dice_coeff, iou_score            : 세그멘테이션 평가 지표
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    모델 출력(logits)을 [0, 1] 확률로 변환.

    - binary (C=1): sigmoid
    - multi-class (C>=2): softmax
    """
    if logits.dim() != 4:
        raise ValueError(f"Expected logits shape (N, C, H, W), got {logits.shape}")

    n, c, h, w = logits.shape
    if c == 1:
        probs = torch.sigmoid(logits)
    else:
        probs = F.softmax(logits, dim=1)
    return probs


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    logits → 바이너리 마스크 (0/1).

    - C=1  : sigmoid 후 threshold
    - C>=2 : softmax 후 argmax==1 (foreground) 기준으로 0/1
    """
    probs = logits_to_probs(logits)  # (N, C, H, W)
    n, c, h, w = probs.shape

    if c == 1:
        mask = (probs > threshold).to(torch.uint8)  # (N, 1, H, W)
        mask = mask[:, 0]                            # (N, H, W)
    else:
        pred_class = probs.argmax(dim=1)             # (N, H, W)
        mask = (pred_class == 1).to(torch.uint8)     # class 1 = foreground 가정

    return mask  # (N, H, W), 0/1


def _flatten_masks(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (N, H, W) 형태의 0/1 마스크 두 개를 (N, H*W)로 펴기.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    # 0/1 보장만 해두고 float로 캐스팅
    pred = pred.reshape(pred.shape[0], -1).float()
    target = target.reshape(target.shape[0], -1).float()
    return pred, target


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice coefficient (binary segmentation 기준).

    pred, target: (N, H, W), 0/1
    return     : 스칼라 (배치 평균 Dice)
    """
    pred, target = _flatten_masks(pred, target)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    IoU (Jaccard index).

    pred, target: (N, H, W), 0/1
    return     : 스칼라 (배치 평균 IoU)
    """
    pred, target = _flatten_masks(pred, target)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()

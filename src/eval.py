# src/eval.py
"""
학습된 U-Net 모델에 대해 validation / test 세트 성능 평가 및 시각화.

- U-Net 논문: 픽셀 단위로 segmentation 성능을 보니
  여기서는 Dice / IoU 두 지표를 사용.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataloader import SegmentationDataset
from model import UNet          # PyTorch 버전이라고 가정
from util import logits_to_mask, dice_coeff, iou_score


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def center_crop_to(
    tensor: torch.Tensor,
    target_hw,
) -> torch.Tensor:
    """
    train.py 와 동일한 중앙 crop 함수.
    padding 없는 conv일 경우, 출력 크기에 맞게 정답/weight를 crop.

    tensor: (N, H, W)
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


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    save_vis_dir: Path | None = None,
    max_vis: int = 5,
):
    """
    전체 데이터셋에 대해 Dice / IoU 평균을 계산하고,
    일부 샘플은 이미지/GT/pred를 그림으로 저장.
    """
    model.eval()
    dices = []
    ious = []

    if save_vis_dir is not None:
        save_vis_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    for idx, batch in enumerate(loader):
        if len(batch) == 3:
            imgs, masks, _ = batch
        else:
            imgs, masks = batch

        imgs = imgs.to(device)           # (N, 1, H, W)
        masks = masks.to(device).long()  # (N, H, W)

        logits = model(imgs)             # (N, C, H_out, W_out)

        # 출력/마스크 크기 맞추기
        if logits.shape[2:] != masks.shape[1:]:
            masks = center_crop_to(masks, logits.shape[2:])

        # 0/1 예측 마스크
        pred_mask = logits_to_mask(logits)        # (N, H, W), 0/1
        true_mask = masks.to(torch.uint8)

        dice = dice_coeff(pred_mask, true_mask).item()
        iou = iou_score(pred_mask, true_mask).item()

        dices.append(dice)
        ious.append(iou)

        # 시각화 몇 장만 저장
        if save_vis_dir is not None and vis_count < max_vis:
            _save_visualization(
                imgs.cpu(),
                true_mask.cpu(),
                pred_mask.cpu(),
                save_vis_dir / f"sample_{idx:03d}.png",
            )
            vis_count += 1

    mean_dice = sum(dices) / len(dices)
    mean_iou = sum(ious) / len(ious)

    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU : {mean_iou:.4f}")

    return mean_dice, mean_iou


def _save_visualization(
    imgs: torch.Tensor,
    true_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    save_path: Path,
):
    """
    이미지 1장에 대해 (원본, GT, Pred) 3개 subplot으로 저장.
    batch size=1 기준으로 작성.
    """
    img = imgs[0, 0].numpy()          # (H, W)
    gt = true_masks[0].numpy()        # (H, W)
    pred = pred_masks[0].numpy()      # (H, W)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT mask")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title("Pred mask")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    # 프로젝트 루트 기준 경로
    project_root = Path(__file__).resolve().parent.parent
    device = get_device()
    print(f"device: {device}")

    # ---------------- 데이터 로더 (여기서는 test 세트 기준) ----------------
    test_dataset = SegmentationDataset(
        image_dir=project_root / "data" / "test" / "images",
        mask_dir=project_root / "data" / "test" / "masks",
        return_weight=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ---------------- 모델 로드 ----------------
    in_channels = 1
    num_classes = 2  # train.py에서 쓴 설정이랑 반드시 맞추기
    model = UNet(in_channels=in_channels, num_classes=num_classes).to(device)

    ckpt_path = project_root / "checkpoints" / "unet_best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    # ---------------- 평가 + 시각화 ----------------
    vis_dir = project_root / "outputs" / "eval_vis"
    evaluate_model(
        model,
        test_loader,
        device,
        save_vis_dir=vis_dir,
        max_vis=5,
    )


if __name__ == "__main__":
    main()

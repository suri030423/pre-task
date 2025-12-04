import os
import cv2
import numpy as np
from glob import glob
import random

# ===== 1) 경로만 네 PC에 맞게 확인 =====
STAGE1_ROOT = r"C:\Users\ITurtle\Desktop\pre-requisite\stage1_train"
OUT_ROOT    = r"C:\Users\ITurtle\Desktop\pre-requisite\data"

train_ratio = 0.7
val_ratio   = 0.15  # 나머지 0.15는 test
random.seed(42)

# ===== 2) 케이스 목록 읽고 train/val/test 나누기 =====
cases = [d for d in os.listdir(STAGE1_ROOT)
         if os.path.isdir(os.path.join(STAGE1_ROOT, d))]
cases = sorted(cases)
random.shuffle(cases)

n_total = len(cases)
n_train = int(n_total * train_ratio)
n_val   = int(n_total * val_ratio)
n_test  = n_total - n_train - n_val

train_cases = cases[:n_train]
val_cases   = cases[n_train:n_train + n_val]
test_cases  = cases[n_train + n_val:]

print(f"총 케이스: {n_total}")
print(f"train: {len(train_cases)}, val: {len(val_cases)}, test: {len(test_cases)}")

splits = {
    "train": train_cases,
    "val":   val_cases,
    "test":  test_cases,
}

# ===== 3) 폴더 생성 =====
def make_dirs(split):
    img_dir = os.path.join(OUT_ROOT, split, "images")
    msk_dir = os.path.join(OUT_ROOT, split, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    return img_dir, msk_dir

# ===== 4) instance 마스크들 → 하나의 binary 마스크로 합치고 저장 =====
def process_case(case_id, split, img_out_dir, msk_out_dir):
    case_dir = os.path.join(STAGE1_ROOT, case_id)
    img_dir  = os.path.join(case_dir, "images")
    msk_dir  = os.path.join(case_dir, "masks")

    img_files = glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.tif"))
    msk_files = glob(os.path.join(msk_dir, "*.png")) + glob(os.path.join(msk_dir, "*.tif"))

    if len(img_files) == 0 or len(msk_files) == 0:
        print(f"[SKIP] {case_id}: image or masks not found")
        return

    image = cv2.imread(img_files[0])

    mask_merged = None
    for mf in msk_files:
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (mask > 0).astype(np.uint8)
        if mask_merged is None:
            mask_merged = mask
        else:
            mask_merged = np.maximum(mask_merged, mask)

    if mask_merged is None:
        print(f"[SKIP] {case_id}: mask load failed")
        return

    fname = f"{case_id}.png"
    cv2.imwrite(os.path.join(img_out_dir, fname), image)
    cv2.imwrite(os.path.join(msk_out_dir, fname), mask_merged * 255)

for split, case_list in splits.items():
    print(f"\n=== {split} ({len(case_list)}) ===")
    img_out, msk_out = make_dirs(split)
    for cid in case_list:
        process_case(cid, split, img_out, msk_out)

print("\n✅ 끝. 이런 구조 생겼을 거임:")
print(OUT_ROOT)

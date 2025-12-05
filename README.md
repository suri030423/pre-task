diff --git a/README.md b/README.md
index e69de29bb2d1d6434b8b29ae775ad8c2e48c5391..311da0478d0b8aaa9d8de5d9c45787b06dfc1142 100644
--- a/README.md
+++ b/README.md
@@ -0,0 +1,44 @@
+# Segmentation U-Net
+
+PyTorch 기반으로 학습/평가할 수 있는 간단한 U-Net 파이프라인입니다.
+
+## 학습 실행
+```bash
+python src/train.py \
+  --root /path/to/project_root \
+  --batch-size 1 \
+  --num-workers 0 \
+  --epochs 50
+```
+
+- 체크포인트는 기본적으로 `<root>/checkpoints/unet_best.pt` 에 저장됩니다.
+- `--checkpoint-dir` 옵션으로 저장 경로를 바꿀 수 있습니다.
+
+## 평가 실행
+```bash
+python src/eval.py \
+  --root /path/to/project_root \
+  --checkpoint /path/to/checkpoints/unet_best.pt \
+  --batch-size 1 \
+  --num-workers 0 \
+  --max-vis 5
+```
+
+- `--max-vis` 값을 0으로 설정하면 시각화 이미지를 저장하지 않습니다.
+- 시각화 이미지는 기본적으로 `<root>/outputs/eval_vis/` 아래에 생성됩니다.


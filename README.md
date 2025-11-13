파일 구조 잡기

대학원 과제/
├─ data/
│  ├─ train/
│  │  ├─ images/
│  │  └─ masks/
│  ├─ val/
│  │  ├─ images/
│  │  └─ masks/
│  └─ test/
│     ├─ images/
│     └─ masks/
│
├─ src/
│  ├─ model.py         # U-Net 구조
│  ├─ dataloader.py    # Kaggle dataset
│  ├─ train.py         # 학습 루프
│  ├─ eval.py          # 평가 / 시각화
│  └─ util.py          # 공용 함수
│
├─ README.md
└─ requirements.txt   (선택)

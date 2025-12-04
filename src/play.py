from dataloader import SegmentationDataset
ds = SegmentationDataset("data/train/images", "data/train/masks")
len(ds)
img, mask = ds[0]
img.shape, img.dtype, mask.shape, mask.dtype

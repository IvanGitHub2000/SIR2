from data_loader import BarcodeDetectionLoader, scale_padded
from dataset_split import split_with_ratio, save_dataset_pairs

IMAGE_SIZE = 640

loader = BarcodeDetectionLoader(IMAGE_SIZE, "D:\\SIR2", lambda img: scale_padded(img, IMAGE_SIZE))

pairs = loader.load_all()

train_pairs, val_pairs = split_with_ratio(pairs, 0.8)

save_dataset_pairs(train_pairs, "D:/SIR2/yolo/train")
save_dataset_pairs(val_pairs, "D:/SIR2/yolo/val")
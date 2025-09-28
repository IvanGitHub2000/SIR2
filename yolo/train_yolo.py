from ultralytics import YOLO
import yaml
import multiprocessing

def main():
    IMAGE_SIZE = 640
    EPOCHS = 30
    BATCH_SIZE = 2

    data_yaml = {
        'train': 'D:/SIR2/yolo/train/images',
        'val': 'D:/SIR2/yolo/val/images',
        'nc': 1,
        'names': ['Barcode']
    }

    with open('train_config.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    model = YOLO("yolov8n.pt")

    # Pokretanje treninga
    results = model.train(
        data='train_config.yaml',
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        verbose=True
    )

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()

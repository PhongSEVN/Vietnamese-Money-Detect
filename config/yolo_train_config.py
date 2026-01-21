DATASET_YOLO = "/home/phong/Desktop/Dataset/Money/Money.v1-version-1.yolov8"

DATA_YAML = DATASET_YOLO+"/data.yaml"

# Training config
YOLO_EPOCHS = 100
YOLO_IMG_SIZE = 640
YOLO_BATCH_SIZE = 4
DEVICE = 0  # GPU

AUGMENTATION = {
    "flipud": 0.3,
    "fliplr": 0.5,
    "degrees": 10,
    "perspective": 0.0001,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "mixup": 0.1,
    "mosaic": 1.0,
    "scale": 0.5
}

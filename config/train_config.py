DATASET_PATH = "/home/phong/Desktop/Dataset/Money/Money.v1-version-1.yolov8"

DATA_YAML = DATASET_PATH+"/data.yaml"

# Training config
EPOCHS = 100
IMGSZ = 640
BATCH = 4
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

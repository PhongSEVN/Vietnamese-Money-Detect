import torch
from ultralytics import YOLO
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.yolo_train_config import DATA_YAML, YOLO_EPOCHS, YOLO_IMG_SIZE, YOLO_BATCH_SIZE


def train_yolo():
    model = YOLO('../yolov8n.pt')

    # single_cls=True: Coi tất cả các class là 1 class (Banknote)
    model.train(
        data=DATA_YAML,
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH_SIZE,
        name='yolo_banknote_det',
        single_cls=True
    )
    model.export(format='onnx')

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU available")
    else:
        print("GPU not available")
    train_yolo()

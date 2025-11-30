from config.train_config import *
from core.trainer import Trainer

import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

if __name__ == "__main__":
    trainer = Trainer("yolov10s.pt")

    trainer.train(
        data_yaml=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        augment=AUGMENTATION
    )

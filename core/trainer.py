from ultralytics import YOLO
from core.utils import say, verify_file

class Trainer:
    def __init__(self, model_path="yolov10s.pt"):
        say("Đang load model YOLOv10...")
        verify_file(model_path)
        self.model = YOLO(model_path)

    def train(self, data_yaml, epochs, imgsz, batch, device, augment):
        verify_file(data_yaml)

        say("Bắt đầu train model...")
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            **augment,
            project="runs_money",
            name="yolov10_train",
            pretrained=True
        )

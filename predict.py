from core.inference import Inference

if __name__ == "__main__":
    model = Inference("runs_money/yolov10_train/weights/best.pt")
    model.run(source=0)  # 0 = webcam

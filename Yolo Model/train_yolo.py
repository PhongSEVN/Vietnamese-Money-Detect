import torch
from ultralytics import YOLO

def train_yolo():
    # Load a model
    model = YOLO('../yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # data: đường dẫn đến file yaml dataset
    # epochs: số vòng lặp (10-20 là đủ demo, 50-100 cho prod)
    # imgsz: kích thước ảnh input
    # single_cls=True: Coi tất cả các class là 1 class (Banknote)
    model.train(
        data=r'd:\IT\Projects\nckh\data\data.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        name='yolo_banknote_det',
        single_cls=True # 1 class
    )

    # Export model
    model.export(format='onnx')

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU available")
    else:
        print("GPU not available")
    train_yolo()

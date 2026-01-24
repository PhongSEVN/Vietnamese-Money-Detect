import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# Import MobileNet model
import sys
sys.path.append(r"D:\IT\Projects\nckh\MobileNet Model")
from mobilenet_model import MobileNet

YOLO_MODEL_PATH = r"D:\IT\Projects\nckh\Yolo Model\runs\detect\yolo_banknote_det3\weights\best.pt"
MOBILENET_MODEL_PATH = r"D:\IT\Projects\nckh\MobileNet Model\trained_mobilenet\best_mobilenet.pt"
CLASS_NAMES = ['100k', '10k', '1k', '200k', '20k', '2k', '500k', '50k', '5k']
CONFIDENCE_THRESHOLD = 0.5


def load_yolo_model(model_path):
    """Load YOLO model để detect vùng có tiền"""
    model = YOLO(model_path)
    print(f"YOLO loaded from: {model_path}")
    return model


def load_mobilenet_model(model_path, device):
    """Load MobileNet model để phân loại mệnh giá"""
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = len(checkpoint.get('class_names', CLASS_NAMES))
    
    model = MobileNet(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    class_names = checkpoint.get('class_names', CLASS_NAMES)
    print(f"MobileNet loaded from: {model_path}")
    return model, class_names


def classify_crop(crop_img, mobilenet, device, class_names):
    """Phân loại một vùng cắt ra từ ảnh"""
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert BGR (OpenCV) to RGB (PIL)
    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    
    input_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = mobilenet(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    return class_names[pred_idx], confidence


def resize_keep_aspect_ratio(image, target_width, target_height):
    """Resize ảnh giữ nguyên tỷ lệ (letterbox)"""
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo canvas màu đen
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Đặt ảnh vào giữa
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    return canvas


def run_camera_test():
    """Chạy test realtime với camera - Fullscreen"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    mobilenet_model, class_names = load_mobilenet_model(MOBILENET_MODEL_PATH, device)
    
    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return
    
    # Tạo window fullscreen
    window_name = "YOLO + MobileNet Money Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO detect vùng có tiền
        results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop vùng tiền
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    # MobileNet phân loại mệnh giá
                    pred_class, cls_conf = classify_crop(crop, mobilenet_model, device, class_names)
                    
                    # Vẽ bounding box và label
                    color = (0, 255, 0)  # Xanh lá
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{pred_class} {cls_conf*100:.1f}%"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Lấy kích thước màn hình
        try:
            _, _, screen_w, screen_h = cv2.getWindowImageRect(window_name)
            if screen_w <= 0 or screen_h <= 0:
                screen_w, screen_h = 1920, 1080
        except:
            screen_w, screen_h = 1920, 1080

        # Resize letterbox
        final_display = resize_keep_aspect_ratio(frame, screen_w, screen_h)

        # Thêm hướng dẫn thoát
        cv2.putText(final_display, "Nhan 'Q' de thoat", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, final_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test ended.")


if __name__ == '__main__':
    run_camera_test()
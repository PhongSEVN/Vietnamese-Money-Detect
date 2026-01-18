import cv2
import numpy as np
import time
import torch
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from verify_core import SafetyVerifier
import os

class RealYOLO:
    def __init__(self, model_path):
        print(f"Loading YOLO from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Chưa có model YOLO tại {model_path}. Hãy chạy train_yolo.py trước!")
        self.model = YOLO(model_path)

    def detect(self, original_image):
        """
        Phát hiện vùng tiền với class 'banknote'.
        Trả về bbox lớn nhất tìm thấy: [x, y, w, h] hoặc None
        """
        # conf=0.5: Ngưỡng tự tin phát hiện
        results = self.model.predict(original_image, conf=0.5, verbose=False)
        
        # Lấy detections
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None
        
        # Tìm box có confidence cao nhất
        best_box = None
        max_conf = -1
        
        for box in boxes:
            conf = float(box.conf)
            if conf > max_conf:
                max_conf = conf
                # Box format xywh (center_x, center_y, w, h) -> convert to xyxy or xywh top-left
                # Ultralytics xywh is center based. xyxy is corner based.
                # Let's get xyxy from box.xyxy and convert to x,y,w,h manually for consistency
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                best_box = [x1, y1, w, h]
                
        return best_box

class RealMobileNet:
    def __init__(self, model_path, class_names):
        print(f"Loading MobileNet from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        
        # Re-init Architecture
        self.model = models.mobilenet_v2(weights=None) # Không cần load weights imagenet ở đây vì sẽ load state_dict
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_names))
        
        # Load weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Chưa có model MobileNet tại {model_path}. Hãy chạy train_mobilenet.py trước!")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Transform giống lúc train
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, roi_image):
        """
        Phân loại mệnh giá từ ảnh ROI (numpy array BGR từ cv2)
        """
        # Convert cv2 (BGR) -> PIL (RGB)
        rgb_img = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
            label = self.class_names[preds.item()]
            confidence = conf.item()
            
        return label, confidence

class MoneyRecognitionSystem:
    def __init__(self):
        print("Đang khởi tạo hệ thống Realtime...")
        
        # Paths to models
        # Note: Ultralytics saves inside 'runs/detect/yolo_banknote_det/weights/best.pt'
        # Em cần check xem file nằm đâu sau khi train xong
        self.yolo_path = r'Yolo Model/runs/detect/yolo_banknote_det2/weights/best.pt'
        self.mobilenet_path = r'mobilenet_banknote.pth'
        
        # 9 classes mệnh giá
        self.class_names = ['100k', '10k', '1k', '200k', '20k', '2k', '500k', '50k', '5k']
        
        # Load Models
        try:
            self.detector = RealYOLO(self.yolo_path)
            self.classifier = RealMobileNet(self.mobilenet_path, self.class_names)
        except FileNotFoundError as e:
            print(f"LỖI: {e}")
            print("Đang chuyển sang chế độ MOCK (Giả lập) để test logic...")
            self.detector = MockYOLO() # Fallback
            self.classifier = MockMobileNet() # Fallback

        self.verifier = SafetyVerifier(history_size=10, blur_threshold=100.0, min_confidence=0.9)
        self.last_audio_time = 0
        self.audio_cooldown = 3.0 

    def speak_result(self, text):
        current_time = time.time()
        if current_time - self.last_audio_time > self.audio_cooldown:
            print(f"🔊 AUDIO OUTPUT: '{text}'")
            self.last_audio_time = current_time
            return True
        return False

    def process_frame(self, frame):
        if frame is None: return
        
        # 1. Detect
        bbox = self.detector.detect(frame)
        
        if bbox is None:
            self.verifier.reset()
            cv2.putText(frame, "Searching...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame

        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 2. Crop with Padding
        padding = 10
        h_img, w_img, _ = frame.shape
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        roi = frame[y1:y2, x1:x2]

        # 3. Verify Quality
        is_valid, reason = self.verifier.check_image_quality(roi)
        if not is_valid:
            cv2.putText(frame, f"Quality Low: {reason}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        # 4. Classify
        label, confidence = self.classifier.predict(roi)
        
        # 5. Verify Temporal
        final_result = self.verifier.verify_prediction(label, confidence)

        debug_text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, debug_text, (x, y + h + 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

        if final_result:
            cv2.putText(frame, f"VERIFIED: {final_result}", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            self.speak_result(final_result)
        else:
            cv2.putText(frame, "Verifying...", (x, y + h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

# --- Classes Giả Lập cho trường hợp chưa train ---
class MockYOLO:
    def detect(self, original_image):
        h, w, _ = original_image.shape
        return [int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4)]

class MockMobileNet:
    def predict(self, roi_image):
        return "500k", 0.95

if __name__ == "__main__":
    # Test với ảnh tĩnh hoặc video
    # cap = cv2.VideoCapture(0)
    
    dummy_frame = cv2.imread(r"C:\Users\as\Desktop\2.jpg") # Cố gắng load 1 ảnh thật nếu có
    if dummy_frame is None:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(dummy_frame, (150, 100), (450, 300), (200, 200, 200), -1) 

    system = MoneyRecognitionSystem()
    
    print("Running system...")
    processed = system.process_frame(dummy_frame)
    cv2.imshow("Result", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Done.")

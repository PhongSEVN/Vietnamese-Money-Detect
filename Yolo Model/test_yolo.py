from ultralytics import YOLO
import cv2
import numpy as np

# Đường dẫn
MODEL_PATH = r"D:\IT\Projects\nckh\Yolo Model\runs\detect\yolo_banknote_det\weights\best.pt"
DATA_YAML = r"D:\IT\Projects\nckh\data\data.yaml"

# Config
YOLO_IMG_SIZE = 640
YOLO_BATCH_SIZE = 4

def test_yolo():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    print("Running validation on test set...")

    # Chạy validation trên test set
    results = model.val(
        data=DATA_YAML,
        split='test',
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH_SIZE,
        verbose=True
    )
    print("KẾT QUẢ TEST:")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")
    print(f"mAP@50:     {results.box.map50:.4f}")
    print(f"mAP@50-95:  {results.box.map:.4f}")

    return results


def predict_single_image(image_path):
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=image_path,
        save=True,
        conf=0.5
    )
    annotated_img = results[0].plot()
    cv2.imshow("YOLO Detection Result", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


def predict_folder(folder_path):
    model = YOLO(MODEL_PATH)
    results = model.predict(
        source=folder_path,
        save=True,
        conf=0.5
    )
    return results

def resize_keep_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Tạo canvas màu đen
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Tính toán vị trí để đặt ảnh vào giữa
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    return canvas

def predict_camera():
    # print("Đang khởi động Camera Fullscreen...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    
    # Đặt tên window để
    window_name = "YOLO Banknote Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Predict
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Vẽ bounding box lên ảnh
        annotated_frame = results[0].plot()
        
        # Lấy kích thước màn hình hiện tại
        # Cách lấy dynamic size của window
        try:
            _, _, screen_w, screen_h = cv2.getWindowImageRect(window_name)
            if screen_w <= 0 or screen_h <= 0: # Fallback nếu chưa lấy được
                screen_w, screen_h = 1920, 1080
        except:
            screen_w, screen_h = 1920, 1080

        # Resize chuẩn tỷ lệ (Letterbox)
        final_display = resize_keep_aspect_ratio(annotated_frame, screen_w, screen_h)

        # Thêm hướng dẫn thoát
        cv2.putText(final_display, "Nhan 'Q' de thoat", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Test trên tập test
    # test_yolo()

    # Test trên một ảnh cụ thể:
    # predict_single_image(r"D:\IT\Projects\nckh\data\test\images\IMG_20251125_221942_jpg.rf.f840972bae356e2b4d3a6c7423a45af4.jpg")

    # Test trên toàn bộ thư mục test:
    # predict_folder(r"D:\IT\Projects\nckh\data\test\images")

    # Test trên camera:
    predict_camera()

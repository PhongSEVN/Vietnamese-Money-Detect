from ultralytics import YOLO
import cv2

# Đường dẫn
MODEL_PATH = r"D:\IT\Projects\nckh\Yolo Model\runs\detect\yolo_banknote_det3\weights\best.pt"
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


if __name__ == '__main__':
    # Test trên tập test
    # test_yolo()
    
    # Test trên một ảnh cụ thể:
    predict_single_image(r"D:\IT\Projects\nckh\data\test\images\IMG_20251125_221942_jpg.rf.f840972bae356e2b4d3a6c7423a45af4.jpg")
    
    # Test trên toàn bộ thư mục test:
    # predict_folder(r"D:\IT\Projects\nckh\data\test\images")

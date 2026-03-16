import os
import cv2
import yaml
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.mobile_train_config import BASE_DATA_PATH, OUTPUT_CLS_PATH


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def denormalize_bbox(x, y, w, h, img_w, img_h):
    x_center = x * img_w
    y_center = y * img_h
    width = w * img_w
    height = h * img_h
    
    x1 = int(max(0, x_center - width/2))
    y1 = int(max(0, y_center - height/2))
    x2 = int(min(img_w, x_center + width/2))
    y2 = int(min(img_h, y_center + height/2))
    return x1, y1, x2, y2

def prepare_classification_data(base_path, output_path):
    """
    Cắt ảnh từ dataset YOLO hiện có để tạo dataset cho MobileNet (Classification).
    """
    yaml_path = os.path.join(base_path, 'data.yaml')
    data_config = load_yaml(yaml_path)
    
    class_names = data_config['names']
    
    # Các tập dữ liệu con
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"Processing {split}...")
        
        # Đường dẫn ảnh và nhãn
        # Theo cấu trúc đã ls: data/train/images và data/train/labels
        img_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"Skipping {split} (path not found)")
            continue
            
        # Tạo thư mục đầu ra
        out_split_dir = os.path.join(output_path, split)
        if not os.path.exists(out_split_dir):
            os.makedirs(out_split_dir)
            
        # Duyệt qua các file ảnh
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(img_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            
            # Đọc nhãn
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                    
                class_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Tính tọa độ cắt
                x1, y1, x2, y2 = denormalize_bbox(cx, cy, bw, bh, w, h)
                
                # Mở rộng vùng cắt một chút (Padding) cho đẹp, giống logic training
                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                # Cắt ảnh
                roi = img[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # Lưu ảnh vào đúng thư mục class
                class_name = class_names[class_id]
                class_dir = os.path.join(out_split_dir, class_name)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                    
                out_name = f"{os.path.splitext(img_file)[0]}_crop_{idx}.jpg"
                cv2.imwrite(os.path.join(class_dir, out_name), roi)

if __name__ == "__main__":
    prepare_classification_data(BASE_DATA_PATH, OUTPUT_CLS_PATH)

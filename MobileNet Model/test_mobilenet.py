import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from mobilenet_model import MobileNet
from dataset.dataset import Money
from config.mobile_train_config import MOBILE_DATA_DIR, MOBILE_IMG_SIZE, NUM_WORKERS

MODEL_PATH = r"D:\IT\Projects\nckh\MobileNet Model\trained_mobilenet\best_mobilenet.pt"

CLASS_NAMES = ['100k', '10k', '1k', '200k', '20k', '2k', '500k', '50k', '5k']


def load_model(model_path, device):
    """Load model đã train"""
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = len(checkpoint.get('class_names', CLASS_NAMES))
    
    model = MobileNet(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    print(f"Best accuracy: {checkpoint.get('best_accuracy', checkpoint.get('accuracy', 'N/A'))}")
    
    return model, checkpoint.get('class_names', CLASS_NAMES)


def test_on_dataset(model, device, class_names):
    """Test model trên toàn bộ test dataset"""
    test_transform = Compose([
        Resize(256),
        CenterCrop(MOBILE_IMG_SIZE),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = Money(MOBILE_DATA_DIR, train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "=" * 50)
    print("KẾT QUẢ TEST:")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    return accuracy


def predict_single_image(model, image_path, device, class_names):
    """Dự đoán trên một ảnh và hiển thị kết quả"""
    transform = Compose([
        Resize(256),
        CenterCrop(MOBILE_IMG_SIZE),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load và transform ảnh
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    pred_class = class_names[pred_idx]
    
    # Hiển thị ảnh với kết quả
    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (400, 400))
    
    label = f"{pred_class}: {confidence*100:.1f}%"
    cv2.putText(img_cv, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("MobileNet Prediction", img_cv)
    print(f"\nDự đoán: {pred_class} (confidence: {confidence*100:.2f}%)")
    print("Nhấn phím bất kỳ để đóng...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return pred_class, confidence


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, class_names = load_model(MODEL_PATH, device)
    
    # Test trên dataset
    test_on_dataset(model, device, class_names)
    
    # Test trên một ảnh cụ thể (uncomment để dùng):
    # predict_single_image(model, r"đường_dẫn_ảnh.jpg", device, class_names)
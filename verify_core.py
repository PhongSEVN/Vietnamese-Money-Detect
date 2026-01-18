import cv2
import numpy as np
from collections import deque
from collections import Counter

class SafetyVerifier:
    def __init__(self, history_size=10, blur_threshold=100.0, min_confidence=0.9):
        """
        Khởi tạo khối Verify.
        
        Args:
            history_size (int): Số lượng frame liên tiếp cần theo dõi để đánh giá nhất quán.
            blur_threshold (float): Ngưỡng để xác định ảnh có bị mờ không (Laplacian variance).
            min_confidence (float): Độ tin cậy tối thiểu để chấp nhận kết quả đơn lẻ.
        """
        self.history_size = history_size
        self.blur_threshold = blur_threshold
        self.min_confidence = min_confidence
        
        # Hàng đợi lưu trữ lịch sử dự đoán: [(label, confidence), ...]
        self.prediction_history = deque(maxlen=history_size)

    def check_image_quality(self, image_roi):
        """
        Kiểm tra chất lượng vùng ảnh (ROI).
        Trả về: (is_valid, reason) - (True/False, Lý do)
        """
        if image_roi is None or image_roi.size == 0:
            return False, "ROI_EMPTY"

        # Chuyển sang ảnh xám để tính toán
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)

        # Kiểm tra độ mờ (Blur check) dùng Laplacian Variance
        # Nếu variance thấp -> ít cạnh -> ảnh mờ
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < self.blur_threshold:
            return False, f"BLURRY (Score: {laplacian_var:.2f})"

        # Kiểm tra độ sáng (Brightness check)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:  # Quá tối
            return False, "TOO_DARK"
        if mean_brightness > 220: # Quá chói
            return False, "TOO_BRIGHT"

        # Kiểm tra kích thước (Size check)
        height, width = gray.shape
        if height < 64 or width < 64: # Ví dụ: nhỏ hơn input của MobileNet (thường 224x224)
            return False, "ROI_TOO_SMALL"

        return True, "OK"

    def verify_prediction(self, current_label, current_confidence):
        """
        Xác thực kết quả dựa trên lịch sử (Temporal Consistency).
        
        Args:
            current_label (str): Mệnh giá vừa dự đoán (ví dụ: '500k').
            current_confidence (float): Độ tin cậy từ MobileNet.
            
        Returns:
            final_result (str or None): Kết quả cuối cùng nếu đủ tin cậy, ngược lại trả về None.
        """
        # Nếu độ tin cậy của frame hiện tại quá thấp, bỏ qua luôn
        if current_confidence < self.min_confidence:
            # Vẫn thêm vào history nhưng đánh dấu là không tin cậy để làm lỏng chuỗi
            self.prediction_history.append(("uncertain", 0.0))
            return None

        # Thêm dự đoán hiện tại vào hàng đợi
        self.prediction_history.append((current_label, current_confidence))

        # Chưa đủ dữ liệu lịch sử
        if len(self.prediction_history) < self.history_size:
            return None

        # Thống kê trong history
        labels = [item[0] for item in self.prediction_history]
        most_common_label, count = Counter(labels).most_common(1)[0]

        # Tiêu chí Verify:
        # Label xuất hiện chiếm đa số (ví dụ > 80% trong buffer)
        consistency_ratio = count / self.history_size
        
        # Nếu là label 'uncertain' thì bỏ qua
        if most_common_label == "uncertain":
            return None

        if consistency_ratio >= 0.8:
            return most_common_label
        else:
            return None

    def reset(self):
        """Reset trạng thái khi bắt đầu phiên mới hoặc mất dấu vật thể quá lâu"""
        self.prediction_history.clear()

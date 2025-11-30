import cv2
from ultralytics import YOLO
from core.utils import DENOMINATION_MAP, calculate_total, say

class Inference:
    def __init__(self, weights):
        self.model = YOLO(weights)
        say("Loaded model for inference")

    def run(self, source=0, conf=0.5):
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Không mở được webcam")
            return

        say("Running webcam with overlay...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict on frame
            results = self.model.predict(frame, conf=conf, verbose=False)

            # Vẽ box lên frame
            annotated_frame = results[0].plot()

            # Tính tổng
            total, detected = calculate_total(results, DENOMINATION_MAP)

            # Overlay tổng tiền
            text = f"Sum: {total} VND"
            cv2.putText(
                annotated_frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            # Hiển thị
            cv2.imshow("Money Counter", annotated_frame)

            # ESC để thoát
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

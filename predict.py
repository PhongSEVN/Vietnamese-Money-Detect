import cv2
from main_system import MoneyRecognitionSystem

if __name__ == "__main__":
    system = MoneyRecognitionSystem()
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = system.process_frame(frame)
        if result is not None:
            cv2.imshow("Banknote Recognition", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

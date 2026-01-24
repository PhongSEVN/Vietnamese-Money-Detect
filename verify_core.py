import cv2
import numpy as np
from collections import deque, Counter
import time

class VerificationStatus:
    """Enum-like class for verification status codes."""
    SEARCHING = "SEARCHING"
    VERIFYING = "VERIFYING"
    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"
    LOW_QUALITY = "LOW_QUALITY"

class SafetyVerifier:
    def __init__(self, history_size=10, blur_threshold=100.0, min_confidence=0.9, stability_threshold=0.8):
        """
        Research-oriented Verification Module.
        
        Args:
            history_size (int): Window size for temporal smoothing.
            blur_threshold (float): Laplacian variance threshold for blur detection.
            min_confidence (float): Minimum softmax probability to consider a single frame valid.
            stability_threshold (float): Ratio of consistent predictions required in history window.
        """
        self.history_size = history_size
        self.blur_threshold = blur_threshold
        self.min_confidence = min_confidence
        self.stability_threshold = stability_threshold
        
        # History buffer: stores tuples of (label, confidence, timestamp)
        self.prediction_history = deque(maxlen=history_size)
        
        # State tracking
        self.current_status = VerificationStatus.SEARCHING
        self.last_confirmed_label = None
        self.consecutive_failures = 0

    def check_image_quality(self, image_roi):
        """
        Evaluates image quality metrics.
        Returns: (is_valid, reason_code, metric_value)
        """
        if image_roi is None or image_roi.size == 0:
            return False, "ROI_EMPTY", 0.0

        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)

        # 1. Blur Detection (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.blur_threshold:
            return False, "BLURRY", laplacian_var

        # 2. Brightness Check
        mean_brightness = np.mean(gray)
        if mean_brightness < 30:
            return False, "TOO_DARK", mean_brightness
        if mean_brightness > 220:
            return False, "TOO_BRIGHT", mean_brightness

        # 3. Resolution Check
        h, w = gray.shape
        if h < 64 or w < 64:
            return False, "LOW_RES", min(h, w)

        return True, "OK", laplacian_var

    def calculate_entropy(self, probs):
        """
        Calculates Shannon entropy of the probability distribution.
        High entropy = High uncertainty (model is confused).
        """
        if probs is None: return 0.0
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs))
        return entropy

    def verify_prediction(self, current_label, current_confidence, full_probs=None):
        """
        Core logic for safety verification.
        
        Args:
            current_label (str): Predicted class.
            current_confidence (float): Top-1 probability.
            full_probs (np.array, optional): Full probability distribution for entropy check.
            
        Returns:
            (status, result_label, debug_info)
        """
        # 1. Instant Rejection based on Confidence
        if current_confidence < self.min_confidence:
            self.prediction_history.append(("uncertain", 0.0, time.time()))
            self.current_status = VerificationStatus.UNCERTAIN
            return VerificationStatus.UNCERTAIN, None, f"Low Conf: {current_confidence:.2f}"

        # 2. Entropy Check (Optional but recommended for research)
        if full_probs is not None:
            entropy = self.calculate_entropy(full_probs)
            # Threshold can be tuned. For 9 classes, max entropy is log2(9) ~ 3.17
            # If entropy > 1.5, the model is quite unsure.
            if entropy > 1.5: 
                self.prediction_history.append(("uncertain", 0.0, time.time()))
                return VerificationStatus.UNCERTAIN, None, f"High Entropy: {entropy:.2f}"

        # 3. Add to History
        self.prediction_history.append((current_label, current_confidence, time.time()))

        # 4. Temporal Consistency Check
        if len(self.prediction_history) < self.history_size:
            self.current_status = VerificationStatus.VERIFYING
            return VerificationStatus.VERIFYING, None, f"Buffering {len(self.prediction_history)}/{self.history_size}"

        # Extract labels ignoring 'uncertain'
        valid_labels = [item[0] for item in self.prediction_history if item[0] != "uncertain"]
        
        if not valid_labels:
            return VerificationStatus.UNCERTAIN, None, "History noisy"

        most_common_label, count = Counter(valid_labels).most_common(1)[0]
        consistency_ratio = count / self.history_size

        # 5. Final Decision
        if consistency_ratio >= self.stability_threshold:
            self.current_status = VerificationStatus.CONFIRMED
            self.last_confirmed_label = most_common_label
            return VerificationStatus.CONFIRMED, most_common_label, f"Stable ({consistency_ratio:.2f})"
        else:
            self.current_status = VerificationStatus.VERIFYING
            return VerificationStatus.VERIFYING, None, f"Unstable ({consistency_ratio:.2f})"

    def reset(self):
        """Resets the verifier state (e.g., when object is lost)."""
        self.prediction_history.clear()
        self.current_status = VerificationStatus.SEARCHING
        self.last_confirmed_label = None

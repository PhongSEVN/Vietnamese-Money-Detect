"""
benchmark_comparison.py
-----------------------
Benchmarks two Vietnamese banknote recognition pipelines:

  Model A (2-stage): YOLO 1-class detector  →  crop  →  MobileNetV3-Large classifier
  Model B (1-stage): YOLO 9-class detector/classifier

Metrics reported
  • Accuracy  : mAP@50 (detection mAP for Model B via ultralytics val;
                         classification mean-AP for Model A via sklearn),
                mAP@50-95 (Model B only, N/A for Model A),
                Precision, Recall, F1 (both models, end-to-end classification),
                per-class breakdown, confusion matrices (PNG)
  • Speed     : end-to-end latency mean ± std over 100 images on CPU (ms/image + FPS)
  • Model size: file size (MB) and parameter count

Output files
  benchmark_results.csv
  benchmark_charts.png
  confusion_matrix_model_a.png
  confusion_matrix_model_b.png
"""

import os, sys, csv, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent

YOLO_1CLASS  = ROOT / "Yolo Model/runs with 1 class/detect/yolo_banknote_det/weights/best.pt"
YOLO_9CLASS  = ROOT / "Yolo Model/runs with 9 classes/detect/yolo_banknote_det/weights/best.pt"
MOBILENET_PT = ROOT / "MobileNet Model/trained_mobilenet/best_mobilenet.pt"
DATA_YAML    = ROOT / "data/data.yaml"
TEST_IMG_DIR = ROOT / "data/test/images"
TEST_LBL_DIR = ROOT / "data/test/labels"

# ── Config (from mobile_train_config.py) ──────────────────────────────────────
CLASSES    = ["100k", "10k", "1k", "200k", "20k", "2k", "500k", "50k", "5k"]
NUM_CLS    = len(CLASSES)
IMG_SIZE   = 224          # MOBILE_IMG_SIZE
SPEED_RUNS = 100
DEVICE     = "cpu"

sys.path.insert(0, str(ROOT / "MobileNet Model"))
from mobilenet_model import MobileNet

MOBILENET_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_mobilenet() -> MobileNet:
    model = MobileNet(num_classes=NUM_CLS, freeze_backbone=False)
    ckpt  = torch.load(MOBILENET_PT, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def parse_test_labels() -> dict:
    out = {}
    for f in TEST_LBL_DIR.glob("*.txt"):
        rows = []
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                rows.append((int(parts[0]), *[float(x) for x in parts[1:]]))
        if rows:
            out[f.stem] = rows
    return out


def collect_image_paths() -> list:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(TEST_IMG_DIR.glob(e))
    return sorted(paths)


def crop_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              padding: int = 10) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w,  x2 + padding)
    y2 = min(h,  y2 + padding)
    return frame[y1:y2, x1:x2]


def classify_crop(crop_bgr: np.ndarray, classifier: MobileNet) -> tuple:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t   = MOBILENET_TF(pil).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(classifier(t), dim=1).squeeze().numpy()
    return int(np.argmax(probs)), probs


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())

def eval_model_b_val() -> dict:
    print("[Model B] Running ultralytics val() …")
    yolo = YOLO(str(YOLO_9CLASS))
    res  = yolo.val(data=str(DATA_YAML), split="test", device=DEVICE,
                    verbose=False, plots=False)
    box = res.box
    p, r = float(box.mp), float(box.mr)
    f1   = 2 * p * r / (p + r + 1e-9)

    # Per-class mAP@50  (box.ap50 → array of length num_classes)
    try:
        ap50_per_cls = list(box.ap50)
    except Exception:
        ap50_per_cls = [0.0] * NUM_CLS

    per_class = {}
    for i, cls in enumerate(CLASSES):
        per_class[cls] = {"AP50": float(ap50_per_cls[i]) if i < len(ap50_per_cls) else 0.0}

    return {
        "mAP50":     float(box.map50),
        "mAP5095":   float(box.map),
        "precision": p,
        "recall":    r,
        "f1":        f1,
        "per_class": per_class,
    }

def run_model_a_predictions(image_paths: list, labels: dict):
    print(f"[Model A] Running {len(image_paths)} images …")
    detector   = YOLO(str(YOLO_1CLASS))
    classifier = load_mobilenet()

    y_true, y_pred = [], []
    y_scores = []

    for idx, p in enumerate(image_paths):
        stem = p.stem
        gt_boxes = labels.get(stem, [])
        if not gt_boxes:
            continue

        gt_cls = gt_boxes[0][0]
        frame  = cv2.imread(str(p))
        if frame is None:
            continue

        det = detector.predict(frame, conf=0.25, device=DEVICE, verbose=False)
        boxes = det[0].boxes

        if len(boxes) == 0:
            y_true.append(gt_cls)
            y_pred.append(-1)
            y_scores.append(np.zeros(NUM_CLS))
            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(image_paths)}")
            continue

        best = int(boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best])
        crop = crop_bbox(frame, x1, y1, x2, y2)

        if crop.size == 0:
            y_true.append(gt_cls)
            y_pred.append(-1)
            y_scores.append(np.zeros(NUM_CLS))
            continue

        pred_cls, probs = classify_crop(crop, classifier)
        y_true.append(gt_cls)
        y_pred.append(pred_cls)
        y_scores.append(probs)

        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(image_paths)}")

    return np.array(y_true), np.array(y_pred), np.array(y_scores)


def run_model_b_predictions(image_paths: list, labels: dict):
    print(f"[Model B] Running {len(image_paths)} images for confusion matrix …")
    yolo = YOLO(str(YOLO_9CLASS))

    y_true, y_pred = [], []

    for idx, p in enumerate(image_paths):
        stem     = p.stem
        gt_boxes = labels.get(stem, [])
        if not gt_boxes:
            continue

        gt_cls = gt_boxes[0][0]
        det    = yolo.predict(str(p), conf=0.25, device=DEVICE, verbose=False)
        boxes  = det[0].boxes

        if len(boxes) == 0:
            y_true.append(gt_cls)
            y_pred.append(-1)
        else:
            best     = int(boxes.conf.argmax())
            pred_cls = int(boxes.cls[best])
            y_true.append(gt_cls)
            y_pred.append(pred_cls)

        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(image_paths)}")

    return np.array(y_true), np.array(y_pred)

def compute_cls_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_scores: np.ndarray | None = None) -> dict:
    valid = y_pred >= 0
    yt    = y_true[valid]
    yp    = y_pred[valid]

    p, r, f1, _ = precision_recall_fscore_support(
        yt, yp, average="macro", zero_division=0, labels=list(range(NUM_CLS))
    )
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        yt, yp, average=None, zero_division=0, labels=list(range(NUM_CLS))
    )

    per_class = {
        CLASSES[i]: {"precision": p_cls[i], "recall": r_cls[i], "f1": f1_cls[i]}
        for i in range(NUM_CLS)
    }

    mean_ap = 0.0
    if y_scores is not None:
        ys = y_scores[valid]
        yt_bin = label_binarize(yt, classes=list(range(NUM_CLS)))
        ap_list = []
        for c in range(NUM_CLS):
            try:
                ap = average_precision_score(yt_bin[:, c], ys[:, c])
                ap_list.append(ap)
            except Exception:
                ap_list.append(0.0)
            per_class[CLASSES[c]]["AP_cls"] = ap_list[-1]
        mean_ap = float(np.mean(ap_list))

    det_rate = float(valid.sum()) / max(len(y_true), 1)

    cm = confusion_matrix(yt, yp, labels=list(range(NUM_CLS)))

    return {
        "mean_AP_cls": mean_ap,
        "precision":   float(p),
        "recall":      float(r),
        "f1":          float(f1),
        "detection_rate": det_rate,
        "per_class":   per_class,
        "cm":          cm,
    }

def _sample(paths: list, n: int) -> list:
    if len(paths) >= n:
        return paths[:n]
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(paths), size=n)
    return [paths[i] for i in idx]


def speed_model_a(image_paths: list, n: int = SPEED_RUNS) -> tuple:
    print(f"[Speed] Model A — {n} images …")
    detector   = YOLO(str(YOLO_1CLASS))
    classifier = load_mobilenet()
    sample     = _sample(image_paths, n)
    lats       = []

    for p in sample:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        t0  = time.perf_counter()
        det = detector.predict(frame, conf=0.25, device=DEVICE, verbose=False)
        boxes = det[0].boxes
        if len(boxes) > 0:
            best = int(boxes.conf.argmax())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best])
            crop = crop_bbox(frame, x1, y1, x2, y2)
            if crop.size > 0:
                classify_crop(crop, classifier)
        lats.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(lats)), float(np.std(lats))


def speed_model_b(image_paths: list, n: int = SPEED_RUNS) -> tuple:
    print(f"[Speed] Model B — {n} images …")
    yolo   = YOLO(str(YOLO_9CLASS))
    sample = _sample(image_paths, n)
    lats   = []

    for p in sample:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        t0 = time.perf_counter()
        yolo.predict(frame, conf=0.25, device=DEVICE, verbose=False)
        lats.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(lats)), float(np.std(lats))

def get_sizes() -> dict:
    print("[Size] Computing model sizes …")

    yolo1 = YOLO(str(YOLO_1CLASS))
    yolo9 = YOLO(str(YOLO_9CLASS))
    mn    = load_mobilenet()

    return {
        "yolo_1cls_mb":    YOLO_1CLASS.stat().st_size  / 1e6,
        "yolo_1cls_params": param_count(yolo1.model),
        "yolo_9cls_mb":    YOLO_9CLASS.stat().st_size  / 1e6,
        "yolo_9cls_params": param_count(yolo9.model),
        "mobilenet_mb":    MOBILENET_PT.stat().st_size / 1e6,
        "mobilenet_params": param_count(mn),
    }

def plot_cm(cm: np.ndarray, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(NUM_CLS))
    ax.set_yticks(range(NUM_CLS))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path.name}")

def plot_charts(ma: dict, mb_val: dict, mb_cls: dict,
                sp_a: tuple, sp_b: tuple, sizes: dict,
                save_path: Path):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model A (2-stage) vs Model B (1-stage) — Vietnamese Banknote",
                 fontsize=14, fontweight="bold")

    COL_A = "#2196F3"   # blue
    COL_B = "#FF9800"   # orange

    ax = axes[0]
    bar_labels = ["Precision", "Recall", "F1", "mean-AP*"]
    a_vals = [ma["precision"], ma["recall"], ma["f1"], ma["mean_AP_cls"]]
    b_vals = [mb_cls["precision"], mb_cls["recall"], mb_cls["f1"], mb_val["mAP50"]]

    x = np.arange(len(bar_labels))
    w = 0.35
    ba = ax.bar(x - w / 2, a_vals, w, label="Model A (2-stage)", color=COL_A, alpha=0.85)
    bb = ax.bar(x + w / 2, b_vals, w, label="Model B (1-stage)", color=COL_B, alpha=0.85)
    ax.set_ylim(0, 1.25)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_title("Accuracy Metrics\n(* cls-AP for A, mAP@50 for B)", fontsize=10)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(list(ba) + list(bb), a_vals + b_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)

    ax = axes[1]
    means = [sp_a[0], sp_b[0]]
    stds  = [sp_a[1], sp_b[1]]
    bars  = ax.bar(["Model A\n(2-stage)", "Model B\n(1-stage)"],
                   means, yerr=stds, color=[COL_A, COL_B],
                   capsize=10, alpha=0.85, width=0.4, error_kw={"elinewidth": 2})
    ax.set_title(f"End-to-End Latency\n({SPEED_RUNS} images, CPU)", fontsize=10)
    ax.set_ylabel("ms / image")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, m, s in zip(bars, means, stds):
        fps = 1000 / m if m > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 1,
                f"{m:.1f}±{s:.1f} ms\n({fps:.1f} FPS)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax = axes[2]
    a_mb = sizes["yolo_1cls_mb"] + sizes["mobilenet_mb"]
    b_mb = sizes["yolo_9cls_mb"]
    a_pm = (sizes["yolo_1cls_params"] + sizes["mobilenet_params"]) / 1e6
    b_pm = sizes["yolo_9cls_params"] / 1e6

    ax.bar(["Model A\n(2-stage)"], [sizes["yolo_1cls_mb"]], color=COL_A, alpha=0.85,
           width=0.4, label="YOLO")
    ax.bar(["Model A\n(2-stage)"], [sizes["mobilenet_mb"]],
           bottom=[sizes["yolo_1cls_mb"]], color=COL_A, alpha=0.5,
           width=0.4, label="MobileNet", hatch="//")
    ax.bar(["Model B\n(1-stage)"], [b_mb], color=COL_B, alpha=0.85, width=0.4)

    ax.set_title("Model File Size (MB)", fontsize=10)
    ax.set_ylabel("MB")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.text(0, a_mb + 0.5, f"{a_mb:.1f} MB\n{a_pm:.1f}M params",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(1, b_mb + 0.5, f"{b_mb:.1f} MB\n{b_pm:.1f}M params",
            ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")

def save_csv(ma: dict, mb_val: dict, mb_cls: dict,
             sp_a: tuple, sp_b: tuple, sizes: dict,
             save_path: Path):
    fps_a = 1000 / sp_a[0] if sp_a[0] > 0 else 0
    fps_b = 1000 / sp_b[0] if sp_b[0] > 0 else 0
    a_total_mb = sizes["yolo_1cls_mb"] + sizes["mobilenet_mb"]
    a_total_p  = sizes["yolo_1cls_params"] + sizes["mobilenet_params"]

    rows = [
        ["BENCHMARK RESULTS — Vietnamese Banknote Recognition"],
        [],
        ["=== OVERALL METRICS ==="],
        ["Metric", "Model A (2-stage)", "Model B (1-stage)", "Notes"],
        ["mAP@50",       f"{ma['mean_AP_cls']:.4f}", f"{mb_val['mAP50']:.4f}",
         "cls-AP for A (sklearn); detection mAP for B (ultralytics)"],
        ["mAP@50-95",    "N/A",                      f"{mb_val['mAP5095']:.4f}",
         "detection metric; not applicable to 2-stage pipeline"],
        ["Precision",    f"{ma['precision']:.4f}",   f"{mb_cls['precision']:.4f}", "end-to-end classification"],
        ["Recall",       f"{ma['recall']:.4f}",      f"{mb_cls['recall']:.4f}",   "end-to-end classification"],
        ["F1",           f"{ma['f1']:.4f}",          f"{mb_cls['f1']:.4f}",       "end-to-end classification"],
        ["Detection rate", f"{ma['detection_rate']:.4f}", f"{mb_cls['detection_rate']:.4f}", "fraction of images with a detection"],
        [],
        ["=== SPEED (CPU, 100 images) ==="],
        ["Metric", "Model A (2-stage)", "Model B (1-stage)", "Unit"],
        ["Latency mean", f"{sp_a[0]:.2f}", f"{sp_b[0]:.2f}", "ms/image"],
        ["Latency std",  f"{sp_a[1]:.2f}", f"{sp_b[1]:.2f}", "ms/image"],
        ["FPS",          f"{fps_a:.1f}",   f"{fps_b:.1f}",   "frames/sec"],
        [],
        ["=== MODEL SIZE ==="],
        ["Component",       "Model A",                      "Model B",                ""],
        ["YOLO file (MB)",  f"{sizes['yolo_1cls_mb']:.2f}", f"{sizes['yolo_9cls_mb']:.2f}", ""],
        ["MobileNet (MB)",  f"{sizes['mobilenet_mb']:.2f}", "—",                      "Model A only"],
        ["Total (MB)",      f"{a_total_mb:.2f}",            f"{sizes['yolo_9cls_mb']:.2f}", ""],
        ["YOLO params",     f"{sizes['yolo_1cls_params']:,}", f"{sizes['yolo_9cls_params']:,}", ""],
        ["MobileNet params",f"{sizes['mobilenet_params']:,}", "—",                     ""],
        ["Total params",    f"{a_total_p:,}",               f"{sizes['yolo_9cls_params']:,}", ""],
        [],
        ["=== PER-CLASS METRICS ==="],
        ["Class",
         "A Prec", "A Recall", "A F1", "A AP_cls",
         "B Prec", "B Recall", "B F1", "B AP@50 (det)"],
    ]

    for cls in CLASSES:
        a_pc = ma["per_class"].get(cls, {})
        b_pc_cls = mb_cls["per_class"].get(cls, {})
        b_pc_val = mb_val["per_class"].get(cls, {})
        rows.append([
            cls,
            f"{a_pc.get('precision', 0):.4f}",
            f"{a_pc.get('recall',    0):.4f}",
            f"{a_pc.get('f1',        0):.4f}",
            f"{a_pc.get('AP_cls',    0):.4f}",
            f"{b_pc_cls.get('precision', 0):.4f}",
            f"{b_pc_cls.get('recall',    0):.4f}",
            f"{b_pc_cls.get('f1',        0):.4f}",
            f"{b_pc_val.get('AP50',      0):.4f}",
        ])

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"  Saved: {save_path.name}")

def print_summary(ma: dict, mb_val: dict, mb_cls: dict,
                  sp_a: tuple, sp_b: tuple, sizes: dict):
    fps_a = 1000 / sp_a[0] if sp_a[0] > 0 else 0
    fps_b = 1000 / sp_b[0] if sp_b[0] > 0 else 0
    a_total_mb = sizes["yolo_1cls_mb"] + sizes["mobilenet_mb"]
    a_total_p  = (sizes["yolo_1cls_params"] + sizes["mobilenet_params"]) / 1e6
    b_total_p  = sizes["yolo_9cls_params"] / 1e6

    SEP = "─" * 68
    DBL = "═" * 68

    print(f"\n{DBL}")
    print("  BENCHMARK: Vietnamese Banknote Recognition".center(68))
    print(f"{DBL}")

    def row(label, a, b):
        print(f"  {label:<26} {str(a):>18} {str(b):>18}")

    print(f"  {'Metric':<26} {'Model A (2-stage)':>18} {'Model B (1-stage)':>18}")
    print(SEP)

    print(f"  {'── Accuracy ──'}")
    row("mAP@50 *",
        f"{ma['mean_AP_cls']:.4f}",
        f"{mb_val['mAP50']:.4f}")
    row("mAP@50-95 *", "N/A", f"{mb_val['mAP5095']:.4f}")
    row("Precision",   f"{ma['precision']:.4f}",  f"{mb_cls['precision']:.4f}")
    row("Recall",      f"{ma['recall']:.4f}",     f"{mb_cls['recall']:.4f}")
    row("F1",          f"{ma['f1']:.4f}",         f"{mb_cls['f1']:.4f}")
    row("Detection rate", f"{ma['detection_rate']:.4f}", f"{mb_cls['detection_rate']:.4f}")
    print(SEP)

    print(f"  {'── Speed (CPU, 100 imgs) ──'}")
    row("Latency mean",  f"{sp_a[0]:.1f} ms",    f"{sp_b[0]:.1f} ms")
    row("Latency std",   f"{sp_a[1]:.1f} ms",    f"{sp_b[1]:.1f} ms")
    row("FPS",           f"{fps_a:.1f}",          f"{fps_b:.1f}")
    print(SEP)

    print(f"  {'── Model Size ──'}")
    row("Total (MB)",    f"{a_total_mb:.1f}",     f"{sizes['yolo_9cls_mb']:.1f}")
    row("Parameters (M)", f"{a_total_p:.2f}",     f"{b_total_p:.2f}")
    row("YOLO (MB)",     f"{sizes['yolo_1cls_mb']:.1f}", f"{sizes['yolo_9cls_mb']:.1f}")
    row("MobileNet (MB)", f"{sizes['mobilenet_mb']:.1f}", "—")
    print(SEP)

    print(f"  {'── Per-Class ──'}")
    print(f"  {'Class':<10} {'A F1':>9} {'A AP_cls':>10} {'B F1':>9} {'B AP@50':>9}")
    for cls in CLASSES:
        a_f1  = ma["per_class"].get(cls, {}).get("f1", 0)
        a_ap  = ma["per_class"].get(cls, {}).get("AP_cls", 0)
        b_f1  = mb_cls["per_class"].get(cls, {}).get("f1", 0)
        b_ap  = mb_val["per_class"].get(cls, {}).get("AP50", 0)
        print(f"  {cls:<10} {a_f1:>9.4f} {a_ap:>10.4f} {b_f1:>9.4f} {b_ap:>9.4f}")

    print(DBL)
    print("  * mAP@50 for Model A = mean classification AP (sklearn);")
    print("    mAP@50 for Model B = detection mAP@IoU0.5 (ultralytics val).")
    print(DBL)

def main():
    print("=" * 68)
    print("  Vietnamese Banknote Benchmark  (CPU mode)")
    print("=" * 68)

    # ── Verify paths ─────────────────────────────────────────────────────────
    for name, path in [("YOLO 1-class", YOLO_1CLASS), ("YOLO 9-class", YOLO_9CLASS),
                       ("MobileNet",    MOBILENET_PT), ("data.yaml",    DATA_YAML),
                       ("test/images",  TEST_IMG_DIR), ("test/labels",  TEST_LBL_DIR)]:
        if not path.exists():
            print(f"  [ERROR] Not found: {name}  →  {path}")
            sys.exit(1)
        print(f"  [OK]  {name:<14}  {path.relative_to(ROOT)}")

    image_paths = collect_image_paths()
    labels      = parse_test_labels()
    print(f"\n  Test images : {len(image_paths)}")
    print(f"  Label files : {len(labels)}")

    print("\n[1/6]  Model B — ultralytics val() …")
    mb_val = eval_model_b_val()

    print("\n[2/6]  Model A — end-to-end predictions …")
    yt_a, yp_a, ys_a = run_model_a_predictions(image_paths, labels)
    ma = compute_cls_metrics(yt_a, yp_a, ys_a)

    print("\n[3/6]  Model B — predictions for confusion matrix …")
    yt_b, yp_b = run_model_b_predictions(image_paths, labels)
    mb_cls = compute_cls_metrics(yt_b, yp_b)

    print("\n[4/6]  Saving confusion matrices …")
    plot_cm(ma["cm"],  "Model A (2-stage) — Confusion Matrix",
            ROOT / "confusion_matrix_model_a.png")
    plot_cm(mb_cls["cm"], "Model B (1-stage) — Confusion Matrix",
            ROOT / "confusion_matrix_model_b.png")

    print(f"\n[5/6]  Speed benchmarks ({SPEED_RUNS} images, CPU) …")
    sp_a = speed_model_a(image_paths)
    sp_b = speed_model_b(image_paths)
    print(f"  Model A: {sp_a[0]:.1f} ± {sp_a[1]:.1f} ms  "
          f"→  {1000/sp_a[0]:.1f} FPS")
    print(f"  Model B: {sp_b[0]:.1f} ± {sp_b[1]:.1f} ms  "
          f"→  {1000/sp_b[0]:.1f} FPS")

    print("\n[6/6]  Model sizes …")
    sizes = get_sizes()

    print("\n  Generating outputs …")
    print_summary(ma, mb_val, mb_cls, sp_a, sp_b, sizes)
    save_csv(ma, mb_val, mb_cls, sp_a, sp_b, sizes, ROOT / "benchmark_results.csv")
    plot_charts(ma, mb_val, mb_cls, sp_a, sp_b, sizes, ROOT / "benchmark_charts.png")

    print("\n  Output files:")
    for f in ["benchmark_results.csv", "benchmark_charts.png",
              "confusion_matrix_model_a.png", "confusion_matrix_model_b.png"]:
        print(f"    {f}")
    print("\n  Done.")


if __name__ == "__main__":
    main()

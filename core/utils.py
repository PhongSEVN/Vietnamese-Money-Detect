import os

def say(msg: str):
    print(f"[UTILS] {msg}")

def verify_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return True

DENOMINATION_MAP = {
    0: 100000,   # '100k'
    1: 10000,    # '10k'
    2: 1000,     # '1k'
    3: 200000,   # '200k'
    4: 20000,    # '20k'
    5: 2000,     # '2k'
    6: 500000,   # '500k'
    7: 50000,    # '50k'
    8: 5000      # '5k'
}


def calculate_total(results, class_to_money):
    total = 0
    detected = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            value = class_to_money.get(cls, 0)
            total += value
            detected.append(value)

    return total, detected

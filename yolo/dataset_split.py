import os
import random


def split_with_ratio(array, ratio):
    if not (0 < ratio < 1):
        raise ValueError("Ratio must be between 0 and 1.")
    random.shuffle(array)
    split_index = int(len(array) * ratio)
    return array[:split_index], array[split_index:]

def parse_annotation(line):
    parts = line.strip().split()
    return float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

def save_dataset_pairs(pairs, folder):
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

    for filepath, image_pil, yolo_bbox, _ in pairs:
        filename = os.path.basename(filepath)
        image_pil.save(os.path.join(folder, "images", filename))
        cx, cy, w, h = [min(max(val, 0), 1) for val in yolo_bbox]
        yolo_label = f"0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}"
        label_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(folder, "labels", label_filename), "w") as f:
            f.write(yolo_label)
            
# corrupt_files = ["20201211_4362_rgb.png"]
        # if filename in corrupt_files:
        #     continue
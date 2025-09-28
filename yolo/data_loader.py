import os
from PIL import Image

def adjust_bounding_box(bbox, new_size, original_image_size, padding):
    cx, cy, w, h = bbox
    original_width, original_height = original_image_size
    padding_x, padding_y = padding

    abs_cx = cx * original_width
    abs_cy = cy * original_height
    abs_w = w * original_width
    abs_h = h * original_height

    scale = min(new_size / original_width, new_size / original_height)
    scale_dim = ((new_size - padding_x) / original_width, (new_size - padding_y) / original_height)

    new_cx = abs_cx * scale
    new_cy = abs_cy * scale

    new_w = abs_w * scale_dim[0] * 1.05
    new_h = abs_h * scale_dim[1] * 1.05

    new_cx /= new_size
    new_cy /= new_size
    new_w /= new_size
    new_h /= new_size

    return new_cx, new_cy, new_w, new_h


def scale_padded(image, desired_size):
    desired_size = [desired_size, desired_size]
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    new_width = desired_size[0]
    new_height = int(new_width / aspect_ratio)

    if new_height > desired_size[1]:
        new_height = desired_size[1]
        new_width = int(new_height * aspect_ratio)

    resized_image = image.resize((new_width, new_height))
    padding_right = abs(new_width - desired_size[0])
    padding_bottom = abs(new_height - desired_size[1])

    padded_image = Image.new("RGB", desired_size, (0, 0, 0))
    padded_image.paste(resized_image, (0, 0))
    return padded_image, (padding_right, padding_bottom)

class BarcodeDetectionLoader:
    def __init__(self, image_size, base_path, preprocess_image):
        self.image_size = image_size
        self.base_path = base_path
        self.preprocess_image = preprocess_image

    def load_annotation(self, annotation_path):
        with open(annotation_path, 'r') as f:
            parts = f.read().strip().split()
        return (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))

    def load_all(self):
        pairs = []
        files_dir = os.path.join(self.base_path, "image_annotations")
        for filename in os.listdir(files_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(files_dir, filename)
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                image, padding = self.preprocess_image(image)
                bbox = self.load_annotation(os.path.join(files_dir, filename.split(".")[0] + ".txt"))
                bbox = adjust_bounding_box(bbox, self.image_size, (width, height), padding)
                pairs.append((image_path, image, bbox, padding))
        return pairs






# class DealKaistLabLoader:
#     def __init__(self, image_size, base_path, preprocess_image):
#         self.image_size = image_size
#         self.base_path = base_path
#         self.preprocess_image = preprocess_image

#     def _load_annotations_dict(self, folder):
#         annotation_path = self.base_path + "/" + folder + ".txt"
#         with open(annotation_path, 'r') as f:
#             lines = f.read().strip().split('\n')

#         annotations = {}
#         for line in lines:
#             segments = line.strip().split(" ")
#             filename = segments[0]
#             x0 = int(segments[1])
#             y0 = int(segments[2])
#             x1 = int(segments[3])
#             y1 = int(segments[4])
#             annotations[filename] = bbox_corner_to_center((x0, y0, x1, y1))
#         return annotations

#     def _load_folder(self, folder):
#         pairs = []
#         annotations_dict = self._load_annotations_dict(folder)
#         image_dir = os.path.join(self.base_path, folder)
#         total = os.listdir(image_dir)

#         for filename in total:
#             if filename.endswith(".jpg") or filename.endswith(".png"):
#                 image_path = os.path.join(image_dir, filename)
#                 image = Image.open(image_path).convert("RGB")
#                 width, height = image.size
#                 image, padding = self.preprocess_image(image)
#                 abs_x, abs_y, abs_width, abs_height = annotations_dict[filename]
#                 bbox = (abs_x / width, abs_y / height, abs_width / width, abs_height / height)
#                 bbox = adjust_bounding_box(bbox, self.image_size, (width, height), padding)
#                 pairs.append((image_path, image, bbox, padding))
#         return pairs

#     def load_all(self):
#         return self._load_folder('rec') + self._load_folder('single_test')

# def bbox_corner_to_center(bbox):
#     xmin, ymin, xmax, ymax = bbox
#     cx = (xmin + xmax) / 2
#     cy = (ymin + ymax) / 2
#     w = xmax - xmin
#     h = ymax - ymin
#     return (cx, cy, w, h)

# def bbox_center_to_corner(bbox):
#     cx, cy, w, h = bbox
#     xmin = cx - w / 2
#     ymin = cy - h / 2
#     xmax = cx + w / 2
#     ymax = cy + h / 2
#     return (xmin, ymin, xmax, ymax)


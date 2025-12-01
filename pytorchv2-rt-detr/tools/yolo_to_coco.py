"""Utility to convert YOLO label files into COCO annotation JSON files.

This helper expects the following directory layout under ``--yolo-root``::

    images/
        train2017/
        val2017/
        ...
    labels/
        train2017/
        val2017/
        ...

Each ``labels/<split>/<image stem>.txt`` should contain standard YOLO lines:

``cls x_center_norm y_center_norm width_norm height_norm``.

The script writes ``instances_<split>.json`` files into ``annotations/`` so
``configs/dataset/coco_detection.yml`` can be used unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image


YOLO_TO_COCO_CATEGORY_ID: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


COCO_CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"},
    {"id": 5, "name": "airplane"},
    {"id": 6, "name": "bus"},
    {"id": 7, "name": "train"},
    {"id": 8, "name": "truck"},
    {"id": 9, "name": "boat"},
    {"id": 10, "name": "traffic light"},
    {"id": 11, "name": "fire hydrant"},
    {"id": 13, "name": "stop sign"},
    {"id": 14, "name": "parking meter"},
    {"id": 15, "name": "bench"},
    {"id": 16, "name": "bird"},
    {"id": 17, "name": "cat"},
    {"id": 18, "name": "dog"},
    {"id": 19, "name": "horse"},
    {"id": 20, "name": "sheep"},
    {"id": 21, "name": "cow"},
    {"id": 22, "name": "elephant"},
    {"id": 23, "name": "bear"},
    {"id": 24, "name": "zebra"},
    {"id": 25, "name": "giraffe"},
    {"id": 27, "name": "backpack"},
    {"id": 28, "name": "umbrella"},
    {"id": 31, "name": "handbag"},
    {"id": 32, "name": "tie"},
    {"id": 33, "name": "suitcase"},
    {"id": 34, "name": "frisbee"},
    {"id": 35, "name": "skis"},
    {"id": 36, "name": "snowboard"},
    {"id": 37, "name": "sports ball"},
    {"id": 38, "name": "kite"},
    {"id": 39, "name": "baseball bat"},
    {"id": 40, "name": "baseball glove"},
    {"id": 41, "name": "skateboard"},
    {"id": 42, "name": "surfboard"},
    {"id": 43, "name": "tennis racket"},
    {"id": 44, "name": "bottle"},
    {"id": 46, "name": "wine glass"},
    {"id": 47, "name": "cup"},
    {"id": 48, "name": "fork"},
    {"id": 49, "name": "knife"},
    {"id": 50, "name": "spoon"},
    {"id": 51, "name": "bowl"},
    {"id": 52, "name": "banana"},
    {"id": 53, "name": "apple"},
    {"id": 54, "name": "sandwich"},
    {"id": 55, "name": "orange"},
    {"id": 56, "name": "broccoli"},
    {"id": 57, "name": "carrot"},
    {"id": 58, "name": "hot dog"},
    {"id": 59, "name": "pizza"},
    {"id": 60, "name": "donut"},
    {"id": 61, "name": "cake"},
    {"id": 62, "name": "chair"},
    {"id": 63, "name": "couch"},
    {"id": 64, "name": "potted plant"},
    {"id": 65, "name": "bed"},
    {"id": 67, "name": "dining table"},
    {"id": 70, "name": "toilet"},
    {"id": 72, "name": "tv"},
    {"id": 73, "name": "laptop"},
    {"id": 74, "name": "mouse"},
    {"id": 75, "name": "remote"},
    {"id": 76, "name": "keyboard"},
    {"id": 77, "name": "cell phone"},
    {"id": 78, "name": "microwave"},
    {"id": 79, "name": "oven"},
    {"id": 80, "name": "toaster"},
    {"id": 81, "name": "sink"},
    {"id": 82, "name": "refrigerator"},
    {"id": 84, "name": "book"},
    {"id": 85, "name": "clock"},
    {"id": 86, "name": "vase"},
    {"id": 87, "name": "scissors"},
    {"id": 88, "name": "teddy bear"},
    {"id": 89, "name": "hair drier"},
    {"id": 90, "name": "toothbrush"},
]


def iter_images(image_dir: Path) -> Iterable[Path]:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        yield from sorted(image_dir.rglob(f"*{suffix}"))


def convert_split(yolo_root: Path, split: str, start_image_id: int = 1) -> dict:
    image_dir = yolo_root / "images" / split
    label_dir = yolo_root / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    images = []
    annotations = []
    ann_id = 1

    for idx, img_path in enumerate(iter_images(image_dir), start=start_image_id):
        with Image.open(img_path) as img:
            width, height = img.size

        images.append(
            {
                "id": idx,
                "file_name": str(img_path.relative_to(image_dir).as_posix()),
                "height": height,
                "width": width,
            }
        )

        label_path = label_dir / img_path.relative_to(image_dir).with_suffix(".txt")
        if not label_path.exists():
            continue

        with label_path.open("r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Unexpected label format in {label_path}: '{line}'")
            cls_idx = int(parts[0])
            if cls_idx < 0 or cls_idx >= len(YOLO_TO_COCO_CATEGORY_ID):
                raise IndexError(
                    f"Class id {cls_idx} in {label_path} is outside the expected 0-{len(YOLO_TO_COCO_CATEGORY_ID) - 1} range"
                )
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            cat_id = YOLO_TO_COCO_CATEGORY_ID[cls_idx]

            abs_w = box_w * width
            abs_h = box_h * height
            abs_x = (x_center * width) - 0.5 * abs_w
            abs_y = (y_center * height) - 0.5 * abs_h

            x0 = max(abs_x, 0.0)
            y0 = max(abs_y, 0.0)
            x1 = min(x0 + abs_w, float(width))
            y1 = min(y0 + abs_h, float(height))
            w = max(x1 - x0, 0.0)
            h = max(y1 - y0, 0.0)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": cat_id,
                    "bbox": [x0, y0, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": COCO_CATEGORIES,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO labels to COCO JSON")
    parser.add_argument("--yolo-root", type=Path, required=True, help="Path to YOLO dataset root")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train2017", "val2017"],
        help="Dataset splits to convert (match directory names under images/ and labels/)",
    )
    parser.add_argument(
        "--start-image-id",
        type=int,
        default=1,
        help="Starting image id (useful when chaining multiple conversions)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yolo_root: Path = args.yolo_root.expanduser().resolve()
    if not yolo_root.exists():
        raise FileNotFoundError(f"YOLO root does not exist: {yolo_root}")

    annotations_dir = yolo_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    next_start_id = args.start_image_id
    for split in args.splits:
        coco_dict = convert_split(yolo_root, split, start_image_id=next_start_id)
        out_file = annotations_dir / f"instances_{split}.json"
        with out_file.open("w") as f:
            json.dump(coco_dict, f)
        print(f"Wrote {out_file} with {len(coco_dict['images'])} images and {len(coco_dict['annotations'])} boxes")

        if coco_dict["images"]:
            next_start_id = coco_dict["images"][-1]["id"] + 1


if __name__ == "__main__":
    main()
"""Prepare VisDrone DET annotations as a one-class YOLO person dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def visdrone_row_to_yolo(row: str, image_width: int, image_height: int, person_categories: set[int]) -> str | None:
    values = [value.strip() for value in row.split(",")]
    if len(values) < 6:
        raise ValueError("Each VisDrone annotation row needs at least six comma-separated values.")
    left, top, width, height = (float(value) for value in values[:4])
    score, category = float(values[4]), int(float(values[5]))
    if score <= 0 or category not in person_categories or width <= 0 or height <= 0:
        return None
    center_x = (left + width / 2) / image_width
    center_y = (top + height / 2) / image_height
    return f"0 {center_x:.6f} {center_y:.6f} {width / image_width:.6f} {height / image_height:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET split annotations to one-class YOLO labels.")
    parser.add_argument("--images", required=True, help="VisDrone images directory.")
    parser.add_argument("--annotations", required=True, help="VisDrone annotation_txt directory.")
    parser.add_argument("--output", required=True, help="Prepared split output directory.")
    parser.add_argument("--person-categories", default="1,2", help="VisDrone categories to map as person (default: pedestrian, people).")
    args = parser.parse_args()
    person_categories = {int(value.strip()) for value in args.person_categories.split(",") if value.strip()}
    source_images, source_annotations, output = Path(args.images), Path(args.annotations), Path(args.output)
    destination_images, destination_labels = output / "images", output / "labels"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    converted = labels = 0
    for image_path in sorted(path for path in source_images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        annotation_path = source_annotations / f"{image_path.stem}.txt"
        if not annotation_path.exists():
            continue
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        converted_rows = [
            label for row in annotation_path.read_text(encoding="utf-8").splitlines()
            if (label := visdrone_row_to_yolo(row, width, height, person_categories)) is not None
        ]
        shutil.copy2(image_path, destination_images / image_path.name)
        (destination_labels / f"{image_path.stem}.txt").write_text("\n".join(converted_rows) + ("\n" if converted_rows else ""), encoding="utf-8")
        converted += 1
        labels += len(converted_rows)
    print({"images": converted, "person_labels": labels, "output": str(output)})


if __name__ == "__main__":
    main()

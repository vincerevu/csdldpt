from __future__ import annotations

import csv
from pathlib import Path

from convert_bmp_to_grayscale import read_bmp_24bit
from search_similar_images import (
    edge_features_from_gray,
    foreground_mask_from_pixels,
    gray_histogram_from_matrix,
    grayscale_matrix,
    shape_features_from_gray,
    texture_features_from_gray,
)


PROJECT_ROOT = Path(__file__).resolve().parent
COLOR_ROOT = PROJECT_ROOT / "data_processed_bmp"
GRAY_ROOT = PROJECT_ROOT / "data_processed_gray_bmp"
OUTPUT_CSV = PROJECT_ROOT / "data_processed_features.csv"

RGB_BINS = 8
GRAY_BINS = 16
SHAPE_COLUMNS = [
    "shape_aspect_ratio",
    "shape_area_ratio",
    "shape_bbox_width",
    "shape_bbox_height",
    "object_center_x",
    "object_center_y",
    "shape_extent",
    "shape_compactness",
]


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total == 0:
        return values
    return [value / total for value in values]


def color_histogram(path: Path) -> list[float]:
    _width, _height, pixels = read_bmp_24bit(path)
    red = [0.0] * RGB_BINS
    green = [0.0] * RGB_BINS
    blue = [0.0] * RGB_BINS
    mask = foreground_mask_from_pixels(pixels)

    for y, row in enumerate(pixels):
        for x, (r, g, b) in enumerate(row):
            if not mask[y][x]:
                continue
            red[min(RGB_BINS - 1, r * RGB_BINS // 256)] += 1
            green[min(RGB_BINS - 1, g * RGB_BINS // 256)] += 1
            blue[min(RGB_BINS - 1, b * RGB_BINS // 256)] += 1

    return normalize(red + green + blue)


def build_features() -> int:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "class_name",
        "image_path",
        "gray_path",
    ]
    fieldnames += [f"color_{index:02d}" for index in range(RGB_BINS * 3)]
    fieldnames += [f"gray_{index:02d}" for index in range(GRAY_BINS)]
    fieldnames += ["edge_density", "edge_bin_0", "edge_bin_45", "edge_bin_90", "edge_bin_135"]
    fieldnames += SHAPE_COLUMNS
    fieldnames += ["texture_horizontal", "texture_vertical", "texture_diagonal", "texture_contrast"]

    count = 0
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in sorted(COLOR_ROOT.rglob("*.bmp")):
            relative = image_path.relative_to(COLOR_ROOT)
            gray_path = GRAY_ROOT / relative
            if not gray_path.exists():
                continue

            color = color_histogram(image_path)
            width, height, color_pixels = read_bmp_24bit(image_path)
            gray_matrix = grayscale_matrix(color_pixels)
            mask = foreground_mask_from_pixels(color_pixels)
            gray = gray_histogram_from_matrix(gray_matrix, mask)
            edge_values = edge_features_from_gray(gray_matrix, width, height, mask)
            shape = shape_features_from_gray(gray_matrix, width, height, mask)
            texture = texture_features_from_gray(gray_matrix, width, height, mask)

            row: dict[str, str] = {
                "image_id": relative.with_suffix("").as_posix().replace("/", "_"),
                "class_name": image_path.parent.name,
                "image_path": str(image_path),
                "gray_path": str(gray_path),
            }

            for index, value in enumerate(color):
                row[f"color_{index:02d}"] = f"{value:.8f}"
            for index, value in enumerate(gray):
                row[f"gray_{index:02d}"] = f"{value:.8f}"

            row["edge_density"] = f"{edge_values[0]:.8f}"
            row["edge_bin_0"] = f"{edge_values[1]:.8f}"
            row["edge_bin_45"] = f"{edge_values[2]:.8f}"
            row["edge_bin_90"] = f"{edge_values[3]:.8f}"
            row["edge_bin_135"] = f"{edge_values[4]:.8f}"

            for column, value in zip(SHAPE_COLUMNS, shape):
                row[column] = f"{value:.8f}"

            row["texture_horizontal"] = f"{texture[0]:.8f}"
            row["texture_vertical"] = f"{texture[1]:.8f}"
            row["texture_diagonal"] = f"{texture[2]:.8f}"
            row["texture_contrast"] = f"{texture[3]:.8f}"

            writer.writerow(row)
            count += 1

    return count


def main() -> None:
    count = build_features()
    print(f"Features written: {count}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

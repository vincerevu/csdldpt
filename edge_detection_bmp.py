from __future__ import annotations

import csv
import math
from pathlib import Path

from convert_bmp_to_grayscale import read_bmp_24bit, write_bmp_24bit


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "data_processed_gray_bmp"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed_edge_bmp"
THRESHOLD = 20
BORDER_MARGIN = 18


def read_gray_matrix(path: Path) -> tuple[int, int, list[list[int]]]:
    width, height, pixels = read_bmp_24bit(path)
    gray: list[list[int]] = []

    for row in pixels:
        gray_row: list[int] = []
        for r, _g, _b in row:
            gray_row.append(r)
        gray.append(gray_row)

    return width, height, gray


def orientation_bin(angle_degrees: float) -> int:
    angle = angle_degrees % 180
    if angle < 22.5 or angle >= 157.5:
        return 0
    if angle < 67.5:
        return 1
    if angle < 112.5:
        return 2
    return 3


def normalize_histogram(values: list[float]) -> list[float]:
    total = sum(values)
    if total == 0:
        return values
    return [value / total for value in values]


def detect_edges(gray: list[list[int]], width: int, height: int) -> tuple[list[list[tuple[int, int, int]]], list[float], int]:
    edge_pixels = [[(0, 0, 0) for _x in range(width)] for _y in range(height)]
    magnitudes, angles = compute_edge_maps(gray, width, height)

    bins = [0.0, 0.0, 0.0, 0.0]
    edge_count = 0
    for y in range(height):
        for x in range(width):
            magnitude = magnitudes[y][x]
            if magnitude <= 0:
                continue
            edge_count += 1
            value = min(255, round(magnitude))
            edge_pixels[y][x] = (value, value, value)
            bins[orientation_bin(angles[y][x])] += magnitude

    return edge_pixels, normalize_histogram(bins), edge_count


def compute_edge_maps(gray: list[list[int]], width: int, height: int) -> tuple[list[list[float]], list[list[float]]]:
    magnitudes = [[0.0 for _x in range(width)] for _y in range(height)]
    angles = [[0.0 for _x in range(width)] for _y in range(height)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if x < BORDER_MARGIN or x >= width - BORDER_MARGIN or y < BORDER_MARGIN or y >= height - BORDER_MARGIN:
                continue

            gx = gray[y][x + 1] - gray[y][x - 1]
            gy = gray[y + 1][x] - gray[y - 1][x]
            magnitude = math.sqrt(gx * gx + gy * gy)

            if magnitude < THRESHOLD:
                continue

            magnitudes[y][x] = magnitude
            angles[y][x] = math.degrees(math.atan2(gy, gx))

    remove_frame_like_lines(magnitudes, width, height)
    return magnitudes, angles


def remove_frame_like_lines(magnitudes: list[list[float]], width: int, height: int) -> None:
    horizontal_limit = max(70, int(width * 0.38))
    vertical_limit = max(70, int(height * 0.38))

    for y in range(height):
        row = magnitudes[y]
        if longest_run(value > 0 for value in row) >= horizontal_limit:
            for x in range(width):
                row[x] = 0.0

    for x in range(width):
        column_has_edge = [magnitudes[y][x] > 0 for y in range(height)]
        if longest_run(column_has_edge) >= vertical_limit:
            for y in range(height):
                magnitudes[y][x] = 0.0


def longest_run(values) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def process_image(source: Path, destination: Path) -> dict[str, str]:
    width, height, gray = read_gray_matrix(source)
    edge_pixels, bins, edge_count = detect_edges(gray, width, height)
    write_bmp_24bit(destination, width, height, edge_pixels)

    total_pixels = width * height
    edge_density = edge_count / total_pixels if total_pixels else 0

    return {
        "class_name": destination.parent.name,
        "source_path": str(source),
        "edge_path": str(destination),
        "width": str(width),
        "height": str(height),
        "threshold": str(THRESHOLD),
        "edge_count": str(edge_count),
        "edge_density": f"{edge_density:.6f}",
        "edge_bin_0": f"{bins[0]:.6f}",
        "edge_bin_45": f"{bins[1]:.6f}",
        "edge_bin_90": f"{bins[2]:.6f}",
        "edge_bin_135": f"{bins[3]:.6f}",
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    skipped: list[str] = []

    for source in sorted(INPUT_ROOT.rglob("*.bmp")):
        relative = source.relative_to(INPUT_ROOT)
        destination = OUTPUT_ROOT / relative

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            rows.append(process_image(source, destination))
        except Exception as exc:
            skipped.append(f"{source}\t{exc}")

    manifest_path = OUTPUT_ROOT / "manifest_edges.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "source_path",
                "edge_path",
                "width",
                "height",
                "threshold",
                "edge_count",
                "edge_density",
                "edge_bin_0",
                "edge_bin_45",
                "edge_bin_90",
                "edge_bin_135",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if skipped:
        (OUTPUT_ROOT / "skipped_edge_files.txt").write_text("\n".join(skipped), encoding="utf-8")

    print(f"Input: {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Converted: {len(rows)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from functools import lru_cache
from pathlib import Path

from convert_bmp_to_grayscale import read_bmp_24bit
from edge_detection_bmp import THRESHOLD, compute_edge_maps, detect_edges, normalize_histogram, orientation_bin


DB_PATH = Path(__file__).resolve().parent / "data_processed_animals.db"
RESULT_DIR = Path(__file__).resolve().parent / "data_processed_search_results"

RGB_BINS = 8
GRAY_BINS = 16

FEATURE_COLUMNS = (
    [f"color_{index:02d}" for index in range(24)]
    + [f"gray_{index:02d}" for index in range(16)]
    + ["edge_density", "edge_bin_0", "edge_bin_45", "edge_bin_90", "edge_bin_135"]
    + [
        "shape_aspect_ratio",
        "shape_area_ratio",
        "shape_bbox_width",
        "shape_bbox_height",
        "object_center_x",
        "object_center_y",
        "shape_extent",
        "shape_compactness",
    ]
    + ["texture_horizontal", "texture_vertical", "texture_diagonal", "texture_contrast"]
)

FEATURE_GROUPS = {
    "color": (0, 24),
    "gray": (24, 40),
    "edge": (40, 45),
    "shape": (45, 53),
    "texture": (53, 57),
}

GROUP_WEIGHTS = {
    "color": 0.05,
    "gray": 0.03,
    "edge": 0.40,
    "shape": 0.30,
    "texture": 0.22,
}

BASE_DISTANCE_WEIGHT = 0.45
LOCAL_GRID_DISTANCE_WEIGHT = 0.55
RERANK_CANDIDATES = 50
LOCAL_GRID_SIZE = 4


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total == 0:
        return values
    return [value / total for value in values]


def foreground_mask_from_pixels(pixels: list[list[tuple[int, int, int]]]) -> list[list[bool]]:
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    mask = [[True for _x in range(width)] for _y in range(height)]

    for y, row in enumerate(pixels):
        for x, (r, g, b) in enumerate(row):
            mask[y][x] = not is_light_background_pixel(r, g, b)

    foreground_count = sum(1 for row in mask for value in row if value)
    total = width * height
    if total == 0 or foreground_count < total * 0.05:
        return [[True for _x in range(width)] for _y in range(height)]
    return mask


def is_light_background_pixel(r: int, g: int, b: int) -> bool:
    bright = r > 232 and g > 232 and b > 232
    nearly_gray = max(r, g, b) - min(r, g, b) < 22
    return bright and nearly_gray


def rgb_to_gray_value(r: int, g: int, b: int) -> int:
    return min(255, max(0, round(0.299 * r + 0.587 * g + 0.114 * b)))


@lru_cache(maxsize=1024)
def local_grid_features_cached(image_path: str) -> tuple[float, ...]:
    return tuple(local_grid_features(Path(image_path)))


def local_grid_features(image_path: Path) -> list[float]:
    width, height, pixels = read_bmp_24bit(image_path)
    x_min = int(width * 0.08)
    x_max = int(width * 0.92)
    y_min = int(height * 0.08)
    y_max = int(height * 0.84)
    features: list[float] = []

    for grid_y in range(LOCAL_GRID_SIZE):
        for grid_x in range(LOCAL_GRID_SIZE):
            start_x = x_min + (x_max - x_min) * grid_x // LOCAL_GRID_SIZE
            end_x = x_min + (x_max - x_min) * (grid_x + 1) // LOCAL_GRID_SIZE
            start_y = y_min + (y_max - y_min) * grid_y // LOCAL_GRID_SIZE
            end_y = y_min + (y_max - y_min) * (grid_y + 1) // LOCAL_GRID_SIZE

            red_sum = 0.0
            green_sum = 0.0
            blue_sum = 0.0
            gray_sum = 0.0
            count = 0

            for y in range(start_y, end_y):
                for x in range(start_x, end_x):
                    r, g, b = pixels[y][x]
                    red_sum += r
                    green_sum += g
                    blue_sum += b
                    gray_sum += rgb_to_gray_value(r, g, b)
                    count += 1

            if count == 0:
                features.extend([0.0, 0.0, 0.0, 0.0])
            else:
                features.extend(
                    [
                        red_sum / count / 255,
                        green_sum / count / 255,
                        blue_sum / count / 255,
                        gray_sum / count / 255,
                    ]
                )

    return features


def color_histogram_from_pixels(pixels: list[list[tuple[int, int, int]]]) -> list[float]:
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


def grayscale_matrix(pixels: list[list[tuple[int, int, int]]]) -> list[list[int]]:
    gray: list[list[int]] = []
    for row in pixels:
        gray.append([rgb_to_gray_value(r, g, b) for r, g, b in row])
    return gray


def gray_histogram_from_matrix(gray: list[list[int]], mask: list[list[bool]] | None = None) -> list[float]:
    bins = [0.0] * GRAY_BINS
    for y, row in enumerate(gray):
        for x, value in enumerate(row):
            if mask is not None and not mask[y][x]:
                continue
            bins[min(GRAY_BINS - 1, value * GRAY_BINS // 256)] += 1
    return normalize(bins)


def edge_features_from_gray(gray: list[list[int]], width: int, height: int, mask: list[list[bool]] | None = None) -> list[float]:
    if mask is None:
        _edge_pixels, bins, edge_count = detect_edges(gray, width, height)
    else:
        magnitudes, angles = compute_edge_maps(gray, width, height)
        bins = [0.0, 0.0, 0.0, 0.0]
        edge_count = 0
        for y in range(height):
            for x in range(width):
                magnitude = magnitudes[y][x]
                if magnitude <= 0 or not mask[y][x]:
                    continue
                edge_count += 1
                bins[orientation_bin(angles[y][x])] += magnitude
        bins = normalize_histogram(bins)
    edge_density = edge_count / (width * height) if width and height else 0
    return [edge_density, bins[0], bins[1], bins[2], bins[3]]


def shape_features_from_gray(
    gray: list[list[int]], width: int, height: int, mask: list[list[bool]] | None = None
) -> list[float]:
    edge_pixels, _bins, edge_count = detect_edges(gray, width, height)
    xs: list[int] = []
    ys: list[int] = []
    masked_edge_count = 0

    for y, row in enumerate(edge_pixels):
        for x, (r, _g, _b) in enumerate(row):
            if r > 0:
                if mask is not None and not mask[y][x]:
                    continue
                xs.append(x)
                ys.append(y)
                masked_edge_count += 1

    if not xs or not ys:
        return [0.0] * 8

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox_area = bbox_width * bbox_height
    image_area = width * height
    bbox_perimeter = 2 * (bbox_width + bbox_height)

    aspect_ratio = bbox_width / bbox_height if bbox_height else 0
    aspect_ratio = min(aspect_ratio / 4, 1)
    area_ratio = bbox_area / image_area if image_area else 0
    bbox_width_ratio = bbox_width / width if width else 0
    bbox_height_ratio = bbox_height / height if height else 0
    center_x = ((min_x + max_x) / 2) / width if width else 0
    center_y = ((min_y + max_y) / 2) / height if height else 0
    extent = masked_edge_count / bbox_area if bbox_area else 0
    compactness = (4 * math.pi * bbox_area) / (bbox_perimeter * bbox_perimeter) if bbox_perimeter else 0

    return [
        aspect_ratio,
        area_ratio,
        bbox_width_ratio,
        bbox_height_ratio,
        center_x,
        center_y,
        extent,
        compactness,
    ]


def texture_features_from_gray(
    gray: list[list[int]], width: int, height: int, mask: list[list[bool]] | None = None
) -> list[float]:
    horizontal_sum = 0.0
    vertical_sum = 0.0
    diagonal_sum = 0.0
    horizontal_count = 0
    vertical_count = 0
    diagonal_count = 0

    for y in range(height):
        for x in range(width):
            current = gray[y][x]
            if x + 1 < width:
                if mask is None or (mask[y][x] and mask[y][x + 1]):
                    horizontal_sum += abs(current - gray[y][x + 1])
                    horizontal_count += 1
            if y + 1 < height:
                if mask is None or (mask[y][x] and mask[y + 1][x]):
                    vertical_sum += abs(current - gray[y + 1][x])
                    vertical_count += 1
            if x + 1 < width and y + 1 < height:
                if mask is None or (mask[y][x] and mask[y + 1][x + 1]):
                    diagonal_sum += abs(current - gray[y + 1][x + 1])
                    diagonal_count += 1

    horizontal = horizontal_sum / horizontal_count / 255 if horizontal_count else 0
    vertical = vertical_sum / vertical_count / 255 if vertical_count else 0
    diagonal = diagonal_sum / diagonal_count / 255 if diagonal_count else 0
    contrast = (horizontal + vertical + diagonal) / 3
    return [horizontal, vertical, diagonal, contrast]


def extract_query_features(image_path: Path) -> list[float]:
    width, height, pixels = read_bmp_24bit(image_path)
    if width != 224 or height != 224:
        raise ValueError("Query image must be a 224x224 uncompressed 24-bit BMP file")

    gray = grayscale_matrix(pixels)
    mask = foreground_mask_from_pixels(pixels)
    color = color_histogram_from_pixels(pixels)
    gray_hist = gray_histogram_from_matrix(gray, mask)
    edge = edge_features_from_gray(gray, width, height, mask)
    shape = shape_features_from_gray(gray, width, height, mask)
    texture = texture_features_from_gray(gray, width, height, mask)
    return color + gray_hist + edge + shape + texture


def euclidean_distance(a: list[float], b: list[float]) -> float:
    total = 0.0
    for left, right in zip(a, b):
        difference = left - right
        total += difference * difference
    return math.sqrt(total)


def histogram_intersection_distance(a: list[float], b: list[float]) -> float:
    return 1 - sum(min(left, right) for left, right in zip(a, b))


def average_l1_distance(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return sum(abs(left - right) for left, right in zip(a, b)) / len(a)


def normalized_l2_distance(a: list[float], b: list[float]) -> float:
    if not a:
        return 0.0
    return euclidean_distance(a, b) / math.sqrt(len(a))


def normalized_l2_distance_tuple(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a:
        return 0.0
    total = 0.0
    for left, right in zip(a, b):
        difference = left - right
        total += difference * difference
    return math.sqrt(total) / math.sqrt(len(a))


def vector_slice(vector: list[float], group_name: str) -> list[float]:
    start, end = FEATURE_GROUPS[group_name]
    return vector[start:end]


def weighted_feature_distance(query_vector: list[float], database_vector: list[float]) -> tuple[float, dict[str, float]]:
    group_distances = {
        "color": histogram_intersection_distance(
            vector_slice(query_vector, "color"),
            vector_slice(database_vector, "color"),
        ),
        "gray": histogram_intersection_distance(
            vector_slice(query_vector, "gray"),
            vector_slice(database_vector, "gray"),
        ),
        "edge": average_l1_distance(
            vector_slice(query_vector, "edge"),
            vector_slice(database_vector, "edge"),
        ),
        "shape": normalized_l2_distance(
            vector_slice(query_vector, "shape"),
            vector_slice(database_vector, "shape"),
        ),
        "texture": normalized_l2_distance(
            vector_slice(query_vector, "texture"),
            vector_slice(database_vector, "texture"),
        ),
    }
    total = sum(GROUP_WEIGHTS[name] * value for name, value in group_distances.items())
    return total, group_distances


def distance_to_similarity(distance: float) -> float:
    return 1 / (1 + distance)


def load_database_features(db_path: Path) -> list[dict[str, object]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        sql = f"""
            SELECT
                i.image_id,
                c.class_name,
                c.vietnamese_name,
                i.file_name,
                i.image_path,
                {", ".join("f." + column for column in FEATURE_COLUMNS)}
            FROM images i
            JOIN animal_classes c ON i.class_id = c.class_id
            JOIN image_features f ON i.image_id = f.image_id
        """
        rows = []
        for row in connection.execute(sql):
            vector = [float(row[column]) for column in FEATURE_COLUMNS]
            rows.append(
                {
                    "image_id": row["image_id"],
                    "class_name": row["class_name"],
                    "vietnamese_name": row["vietnamese_name"],
                    "file_name": row["file_name"],
                    "image_path": row["image_path"],
                    "vector": vector,
                }
            )
        return rows
    finally:
        connection.close()


def search(query_image: Path, top_k: int, db_path: Path, exclude_self: bool) -> tuple[list[float], list[dict[str, object]]]:
    query_vector = extract_query_features(query_image)
    database_rows = load_database_features(db_path)
    query_resolved = str(query_image.resolve()).lower()
    query_grid = local_grid_features_cached(str(query_image.resolve()))

    results: list[dict[str, object]] = []
    for row in database_rows:
        row_path = str(Path(str(row["image_path"])).resolve()).lower()
        if exclude_self and row_path == query_resolved:
            continue
        distance, group_distances = weighted_feature_distance(query_vector, row["vector"])  # type: ignore[arg-type]
        similarity = distance_to_similarity(distance)
        results.append(
            {
                **row,
                "base_distance": distance,
                "distance": distance,
                "similarity": similarity,
                "group_distances": group_distances,
            }
        )

    results.sort(key=lambda item: item["base_distance"])
    candidates = results[: max(top_k, RERANK_CANDIDATES)]

    for item in candidates:
        item_grid = local_grid_features_cached(str(Path(str(item["image_path"])).resolve()))
        local_distance = normalized_l2_distance_tuple(query_grid, item_grid)
        distance = BASE_DISTANCE_WEIGHT * float(item["base_distance"]) + LOCAL_GRID_DISTANCE_WEIGHT * local_distance
        item["local_grid_distance"] = local_distance
        item["distance"] = distance
        item["similarity"] = distance_to_similarity(distance)

    candidates.sort(key=lambda item: item["distance"])
    return query_vector, candidates[:top_k]


def export_results(query_image: Path, query_vector: list[float], results: list[dict[str, object]]) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / f"{query_image.stem}_top5.csv"

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "query_image",
                "feature_count",
                "edge_threshold",
                "rank",
                "image_id",
                "class_name",
                "vietnamese_name",
                "distance",
                "similarity",
                "base_distance",
                "local_grid_distance",
                "color_distance",
                "gray_distance",
                "edge_distance",
                "shape_distance",
                "texture_distance",
                "image_path",
            ]
        )
        for rank, result in enumerate(results, start=1):
            group_distances = result.get("group_distances", {})
            writer.writerow(
                [
                    str(query_image),
                    len(query_vector),
                    THRESHOLD,
                    rank,
                    result["image_id"],
                    result["class_name"],
                    result["vietnamese_name"],
                    f"{result['distance']:.8f}",
                    f"{result['similarity']:.8f}",
                    f"{result.get('base_distance', 0):.8f}",
                    f"{result.get('local_grid_distance', 0):.8f}",
                    f"{group_distances.get('color', 0):.8f}",
                    f"{group_distances.get('gray', 0):.8f}",
                    f"{group_distances.get('edge', 0):.8f}",
                    f"{group_distances.get('shape', 0):.8f}",
                    f"{group_distances.get('texture', 0):.8f}",
                    result["image_path"],
                ]
            )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search top-5 similar quadruped animal BMP images.")
    parser.add_argument("query_image", type=Path, help="Path to a 224x224 uncompressed 24-bit BMP image.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--exclude-self", action="store_true", help="Ignore the query image if it already exists in the database.")
    args = parser.parse_args()

    query_vector, results = search(args.query_image, args.top_k, args.db, args.exclude_self)
    output_path = export_results(args.query_image, query_vector, results)

    print(f"Query image: {args.query_image}")
    print(f"Feature count: {len(query_vector)}")
    print("Top similar images:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['file_name']} | {result['class_name']} | "
            f"distance={result['distance']:.6f} | similarity={result['similarity']:.6f} | "
            f"{result['image_path']}"
        )
    print(f"Intermediate result: {output_path}")


if __name__ == "__main__":
    main()

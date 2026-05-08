from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

from search_similar_images import (
    BASE_DISTANCE_WEIGHT,
    FEATURE_COLUMNS,
    LOCAL_GRID_DISTANCE_WEIGHT,
    RERANK_CANDIDATES,
    distance_to_similarity,
    local_grid_features_cached,
    normalized_l2_distance_tuple,
    weighted_feature_distance,
)


DB_PATH = Path(__file__).resolve().parent / "data_processed_animals.db"
OUTPUT_CSV = Path(__file__).resolve().parent / "data_processed_evaluation_precision_at_5.csv"
TOP_K = 5


def load_rows() -> list[dict[str, object]]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        sql = f"""
            SELECT
                i.image_id,
                i.file_name,
                c.class_name,
                i.image_path,
                {", ".join("f." + column for column in FEATURE_COLUMNS)}
            FROM images i
            JOIN animal_classes c ON i.class_id = c.class_id
            JOIN image_features f ON i.image_id = f.image_id
            ORDER BY i.image_id
        """
        rows = []
        for row in connection.execute(sql):
            rows.append(
                {
                    "image_id": row["image_id"],
                    "file_name": row["file_name"],
                    "class_name": row["class_name"],
                    "image_path": row["image_path"],
                    "vector": [float(row[column]) for column in FEATURE_COLUMNS],
                }
            )
        return rows
    finally:
        connection.close()


def top_k_for_query(query: dict[str, object], rows: list[dict[str, object]]) -> list[dict[str, object]]:
    results = []
    query_vector = query["vector"]
    query_grid = local_grid_features_cached(str(Path(str(query["image_path"])).resolve()))

    for row in rows:
        if row["image_id"] == query["image_id"]:
            continue
        distance, _group_distances = weighted_feature_distance(query_vector, row["vector"])  # type: ignore[arg-type]
        results.append({**row, "base_distance": distance})

    results.sort(key=lambda item: item["base_distance"])
    candidates = results[: max(TOP_K, RERANK_CANDIDATES)]

    for item in candidates:
        item_grid = local_grid_features_cached(str(Path(str(item["image_path"])).resolve()))
        local_distance = normalized_l2_distance_tuple(query_grid, item_grid)
        distance = BASE_DISTANCE_WEIGHT * float(item["base_distance"]) + LOCAL_GRID_DISTANCE_WEIGHT * local_distance
        item["local_grid_distance"] = local_distance
        item["distance"] = distance
        item["similarity"] = distance_to_similarity(distance)

    candidates.sort(key=lambda item: item["distance"])
    return candidates[:TOP_K]


def main() -> None:
    rows = load_rows()
    if not rows:
        raise RuntimeError("No indexed images found in animals.db")

    summary: dict[str, list[float]] = defaultdict(list)
    top1_hits = 0
    precision_values = []

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "query_image_id",
                "query_class",
                "top1_class",
                "top1_hit",
                "correct_in_top5",
                "precision_at_5",
                "top5_image_ids",
                "top5_classes",
                "top5_distances",
            ]
        )

        for query in rows:
            top_k = top_k_for_query(query, rows)
            query_class = str(query["class_name"])
            correct = sum(1 for item in top_k if item["class_name"] == query_class)
            precision = correct / TOP_K
            top1_hit = 1 if top_k and top_k[0]["class_name"] == query_class else 0

            top1_hits += top1_hit
            precision_values.append(precision)
            summary[query_class].append(precision)

            writer.writerow(
                [
                    query["image_id"],
                    query_class,
                    top_k[0]["class_name"] if top_k else "",
                    top1_hit,
                    correct,
                    f"{precision:.4f}",
                    ";".join(str(item["image_id"]) for item in top_k),
                    ";".join(str(item["class_name"]) for item in top_k),
                    ";".join(f"{item['distance']:.6f}" for item in top_k),
                ]
            )

    mean_precision = sum(precision_values) / len(precision_values)
    top1_accuracy = top1_hits / len(rows)

    print(f"Images evaluated: {len(rows)}")
    print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    print(f"Mean Precision@5: {mean_precision:.4f}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print()
    print("Per-class Precision@5:")
    for class_name in sorted(summary):
        values = summary[class_name]
        print(f"{class_name}: {sum(values) / len(values):.4f} ({len(values)} images)")


if __name__ == "__main__":
    main()

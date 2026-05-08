from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_CSV = PROJECT_ROOT / "data_processed_features.csv"
DB_PATH = PROJECT_ROOT / "data_processed_animals.db"

CLASS_VI_NAMES = {
    "alpaca": "Lạc đà Alpaca",
    "bear": "Gấu",
    "buffalo": "Trâu",
    "cat": "Mèo",
    "camel": "Lạc đà",
    "cow": "Bò",
    "deer": "Hươu",
    "coyote": "Chó sói đồng cỏ",
    "dog": "Chó",
    "elephant": "Voi",
    "fox": "Cáo",
    "giraffe": "Hươu cao cổ",
    "goat": "Dê",
    "hippo": "Hà mã",
    "hyena": "Linh cẩu",
    "jackal": "Chó rừng",
    "jaguar": "Báo đốm",
    "leopard": "Báo",
    "lion": "Sư tử",
    "lynx": "Linh miêu",
    "mongoose": "Cầy mangut",
    "moose": "Nai sừng tấm",
    "panda": "Gấu trúc",
    "pig": "Heo",
    "rabbit": "Thỏ",
    "reindeer": "Tuần lộc",
    "rhino": "Tê giác",
    "sheep": "Cừu",
    "snow_leopard": "Báo tuyết",
    "tiger": "Hổ",
    "white_tiger": "Hổ trắng",
    "wildcat": "Mèo rừng",
    "wolf": "Sói",
    "zebra": "Ngựa vằn",
}

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


def create_schema(connection: sqlite3.Connection) -> None:
    feature_columns_sql = ",\n        ".join(f"{column} REAL NOT NULL" for column in FEATURE_COLUMNS)
    connection.executescript(
        f"""
        PRAGMA foreign_keys = ON;

        DROP TABLE IF EXISTS image_features;
        DROP TABLE IF EXISTS images;
        DROP TABLE IF EXISTS animal_classes;

        CREATE TABLE animal_classes (
            class_id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT NOT NULL UNIQUE,
            vietnamese_name TEXT NOT NULL,
            description TEXT
        );

        CREATE TABLE images (
            image_id TEXT PRIMARY KEY,
            class_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            image_path TEXT NOT NULL UNIQUE,
            gray_path TEXT NOT NULL,
            width INTEGER NOT NULL DEFAULT 224,
            height INTEGER NOT NULL DEFAULT 224,
            format TEXT NOT NULL DEFAULT 'BMP',
            FOREIGN KEY (class_id) REFERENCES animal_classes(class_id)
        );

        CREATE TABLE image_features (
            image_id TEXT PRIMARY KEY,
            {feature_columns_sql},
            FOREIGN KEY (image_id) REFERENCES images(image_id)
        );

        CREATE INDEX idx_images_class_id ON images(class_id);
        """
    )


def insert_classes(connection: sqlite3.Connection, rows: list[dict[str, str]]) -> dict[str, int]:
    class_names = sorted({row["class_name"] for row in rows})
    for class_name in class_names:
        vietnamese_name = CLASS_VI_NAMES.get(class_name, class_name)
        description = f"Nhóm ảnh động vật bốn chân: {vietnamese_name}"
        connection.execute(
            """
            INSERT INTO animal_classes (class_name, vietnamese_name, description)
            VALUES (?, ?, ?)
            """,
            (class_name, vietnamese_name, description),
        )

    result: dict[str, int] = {}
    for row in connection.execute("SELECT class_id, class_name FROM animal_classes"):
        result[row[1]] = row[0]
    return result


def insert_images_and_features(connection: sqlite3.Connection, rows: list[dict[str, str]], class_ids: dict[str, int]) -> None:
    feature_placeholders = ", ".join("?" for _ in FEATURE_COLUMNS)
    feature_column_sql = ", ".join(FEATURE_COLUMNS)

    for row in rows:
        image_path = Path(row["image_path"])
        connection.execute(
            """
            INSERT INTO images (
                image_id, class_id, file_name, image_path, gray_path, width, height, format
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["image_id"],
                class_ids[row["class_name"]],
                image_path.name,
                row["image_path"],
                row["gray_path"],
                224,
                224,
                "BMP",
            ),
        )

        values = [float(row[column]) for column in FEATURE_COLUMNS]
        connection.execute(
            f"""
            INSERT INTO image_features (image_id, {feature_column_sql})
            VALUES (?, {feature_placeholders})
            """,
            [row["image_id"], *values],
        )


def build_database() -> None:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_CSV}")

    with FEATURES_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    try:
        create_schema(connection)
        class_ids = insert_classes(connection, rows)
        insert_images_and_features(connection, rows, class_ids)
        connection.commit()
    finally:
        connection.close()

    print(f"Database: {DB_PATH}")
    print(f"Classes: {len(class_ids)}")
    print(f"Images: {len(rows)}")
    print(f"Features per image: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    build_database()

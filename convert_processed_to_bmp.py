from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "data_processed"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed_bmp"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    skipped: list[str] = []

    for source in sorted(INPUT_ROOT.rglob("*.jpg")):
        if source.name.lower() == "manifest.csv":
            continue

        relative = source.relative_to(INPUT_ROOT)
        destination = OUTPUT_ROOT / relative.with_suffix(".bmp")
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.save(destination, "BMP")
                width, height = image.size
            rows.append(
                {
                    "class_name": destination.parent.name,
                    "source_path": str(source),
                    "bmp_path": str(destination),
                    "width": str(width),
                    "height": str(height),
                    "format": "BMP_24BIT_RGB",
                }
            )
        except Exception as exc:
            skipped.append(f"{source}\t{exc}")

    manifest_path = OUTPUT_ROOT / "manifest_bmp.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_name", "source_path", "bmp_path", "width", "height", "format"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if skipped:
        (OUTPUT_ROOT / "skipped_bmp_files.txt").write_text("\n".join(skipped), encoding="utf-8")

    print(f"Input: {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Converted: {len(rows)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

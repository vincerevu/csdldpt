from __future__ import annotations

import csv
import struct
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "data_processed_bmp"
OUTPUT_ROOT = PROJECT_ROOT / "data_processed_gray_bmp"


def read_bmp_24bit(path: Path) -> tuple[int, int, list[list[tuple[int, int, int]]]]:
    data = path.read_bytes()

    if data[0:2] != b"BM":
        raise ValueError("Not a BMP file")

    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    dib_header_size = struct.unpack_from("<I", data, 14)[0]
    if dib_header_size < 40:
        raise ValueError("Unsupported BMP DIB header")

    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
    compression = struct.unpack_from("<I", data, 30)[0]

    if planes != 1 or bits_per_pixel != 24 or compression != 0:
        raise ValueError("Only uncompressed 24-bit BMP is supported")

    abs_height = abs(height)
    row_size = ((width * 3 + 3) // 4) * 4
    pixels: list[list[tuple[int, int, int]]] = []

    for row in range(abs_height):
        source_row = abs_height - 1 - row if height > 0 else row
        row_start = pixel_offset + source_row * row_size
        pixel_row: list[tuple[int, int, int]] = []
        for col in range(width):
            offset = row_start + col * 3
            b = data[offset]
            g = data[offset + 1]
            r = data[offset + 2]
            pixel_row.append((r, g, b))
        pixels.append(pixel_row)

    return width, abs_height, pixels


def rgb_to_gray(r: int, g: int, b: int) -> int:
    return min(255, max(0, round(0.299 * r + 0.587 * g + 0.114 * b)))


def write_bmp_24bit(path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> None:
    row_size = ((width * 3 + 3) // 4) * 4
    pixel_array_size = row_size * height
    file_size = 14 + 40 + pixel_array_size

    header = bytearray()
    header.extend(b"BM")
    header.extend(struct.pack("<I", file_size))
    header.extend(struct.pack("<HH", 0, 0))
    header.extend(struct.pack("<I", 14 + 40))

    dib = bytearray()
    dib.extend(struct.pack("<I", 40))
    dib.extend(struct.pack("<i", width))
    dib.extend(struct.pack("<i", height))
    dib.extend(struct.pack("<H", 1))
    dib.extend(struct.pack("<H", 24))
    dib.extend(struct.pack("<I", 0))
    dib.extend(struct.pack("<I", pixel_array_size))
    dib.extend(struct.pack("<i", 2835))
    dib.extend(struct.pack("<i", 2835))
    dib.extend(struct.pack("<I", 0))
    dib.extend(struct.pack("<I", 0))

    output = bytearray(header + dib)
    padding = bytes(row_size - width * 3)

    for row in range(height - 1, -1, -1):
        for col in range(width):
            r, g, b = pixels[row][col]
            output.extend(bytes((b, g, r)))
        output.extend(padding)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(output)


def convert_to_grayscale(source: Path, destination: Path) -> tuple[int, int]:
    width, height, pixels = read_bmp_24bit(source)
    gray_pixels: list[list[tuple[int, int, int]]] = []

    for row in pixels:
        gray_row: list[tuple[int, int, int]] = []
        for r, g, b in row:
            gray = rgb_to_gray(r, g, b)
            gray_row.append((gray, gray, gray))
        gray_pixels.append(gray_row)

    write_bmp_24bit(destination, width, height, gray_pixels)
    return width, height


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    skipped: list[str] = []

    for source in sorted(INPUT_ROOT.rglob("*.bmp")):
        relative = source.relative_to(INPUT_ROOT)
        destination = OUTPUT_ROOT / relative

        try:
            width, height = convert_to_grayscale(source, destination)
            rows.append(
                {
                    "class_name": destination.parent.name,
                    "source_path": str(source),
                    "gray_path": str(destination),
                    "width": str(width),
                    "height": str(height),
                    "formula": "gray=0.299R+0.587G+0.114B",
                }
            )
        except Exception as exc:
            skipped.append(f"{source}\t{exc}")

    manifest_path = OUTPUT_ROOT / "manifest_gray.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["class_name", "source_path", "gray_path", "width", "height", "formula"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if skipped:
        (OUTPUT_ROOT / "skipped_gray_files.txt").write_text("\n".join(skipped), encoding="utf-8")

    print(f"Input: {INPUT_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Converted: {len(rows)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import streamlit as st

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - Streamlit demo can still use BMP files.
    Image = None
    ImageOps = None

from convert_bmp_to_grayscale import read_bmp_24bit, rgb_to_gray, write_bmp_24bit
from edge_detection_bmp import THRESHOLD, detect_edges
from search_similar_images import (
    DB_PATH,
    FEATURE_COLUMNS,
    GROUP_WEIGHTS,
    BASE_DISTANCE_WEIGHT,
    LOCAL_GRID_DISTANCE_WEIGHT,
    distance_to_similarity,
    extract_query_features,
    load_database_features,
    search,
    weighted_feature_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data_processed_bmp"
TMP_ROOT = PROJECT_ROOT / ".streamlit_demo_tmp"
IMAGE_SIZE = 224

FEATURE_GROUPS = {
    "Màu sắc RGB histogram": FEATURE_COLUMNS[0:24],
    "Mức xám histogram": FEATURE_COLUMNS[24:40],
    "Biên / hướng cạnh": FEATURE_COLUMNS[40:45],
    "Hình dạng / kích thước": FEATURE_COLUMNS[45:53],
    "Kết cấu": FEATURE_COLUMNS[53:57],
}


def page_config() -> None:
    st.set_page_config(
        page_title="Demo CBIR động vật bốn chân",
        layout="wide",
    )


@st.cache_data(show_spinner=False)
def load_dataset_summary() -> list[dict[str, object]]:
    if not DB_PATH.exists():
        return []

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT c.class_name, c.vietnamese_name, COUNT(i.image_id) AS image_count
            FROM animal_classes c
            LEFT JOIN images i ON i.class_id = c.class_id
            GROUP BY c.class_id, c.class_name, c.vietnamese_name
            ORDER BY c.class_name
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


@st.cache_data(show_spinner=False)
def list_dataset_images() -> dict[str, list[str]]:
    images: dict[str, list[str]] = {}
    if not DATASET_ROOT.exists():
        return images

    for class_dir in sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir()):
        files = sorted(str(path) for path in class_dir.glob("*.bmp"))
        if files:
            images[class_dir.name] = files
    return images


def prepare_uploaded_image(uploaded_file) -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower()
    source_path = TMP_ROOT / f"query_{uuid.uuid4().hex}{suffix}"
    source_path.write_bytes(uploaded_file.getbuffer())

    if suffix == ".bmp":
        width, height, _pixels = read_bmp_24bit(source_path)
        if width == IMAGE_SIZE and height == IMAGE_SIZE:
            return source_path

    if Image is None or ImageOps is None:
        raise RuntimeError("Cần cài Pillow để upload JPG/PNG hoặc BMP chưa đúng 224x224.")

    image = Image.open(source_path).convert("RGB")
    image = ImageOps.contain(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (255, 255, 255))
    left = (IMAGE_SIZE - image.width) // 2
    top = (IMAGE_SIZE - image.height) // 2
    canvas.paste(image, (left, top))

    bmp_path = source_path.with_suffix(".bmp")
    canvas.save(bmp_path, format="BMP")
    return bmp_path


def create_intermediate_images(query_path: Path) -> tuple[Path, Path]:
    width, height, pixels = read_bmp_24bit(query_path)
    gray_matrix: list[list[int]] = []
    gray_pixels: list[list[tuple[int, int, int]]] = []

    for row in pixels:
        gray_row: list[int] = []
        gray_pixel_row: list[tuple[int, int, int]] = []
        for r, g, b in row:
            value = rgb_to_gray(r, g, b)
            gray_row.append(value)
            gray_pixel_row.append((value, value, value))
        gray_matrix.append(gray_row)
        gray_pixels.append(gray_pixel_row)

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    gray_path = TMP_ROOT / f"{query_path.stem}_gray.bmp"
    edge_path = TMP_ROOT / f"{query_path.stem}_edge.bmp"
    write_bmp_24bit(gray_path, width, height, gray_pixels)

    edge_pixels, _bins, _edge_count = detect_edges(gray_matrix, width, height)
    write_bmp_24bit(edge_path, width, height, edge_pixels)
    return gray_path, edge_path


def feature_rows(vector: list[float], columns: list[str]) -> list[dict[str, object]]:
    return [
        {"Thuộc tính": column, "Giá trị": round(vector[FEATURE_COLUMNS.index(column)], 8)}
        for column in columns
    ]


def render_sidebar() -> tuple[Path | None, bool]:
    st.sidebar.header("Ảnh truy vấn")
    mode = st.sidebar.radio("Nguồn ảnh", ["Chọn từ dataset", "Upload ảnh mới"], index=0)
    exclude_self = st.sidebar.checkbox("Bỏ qua chính ảnh truy vấn nếu có trong CSDL", value=True)

    if mode == "Upload ảnh mới":
        uploaded = st.sidebar.file_uploader("Chọn ảnh BMP/JPG/PNG", type=["bmp", "jpg", "jpeg", "png"])
        if uploaded is None:
            return None, exclude_self
        try:
            return prepare_uploaded_image(uploaded), exclude_self
        except Exception as exc:
            st.sidebar.error(str(exc))
            return None, exclude_self

    images_by_class = list_dataset_images()
    if not images_by_class:
        st.sidebar.error("Không tìm thấy dataset BMP.")
        return None, exclude_self

    class_name = st.sidebar.selectbox("Loài", list(images_by_class.keys()))
    image_paths = images_by_class[class_name]
    image_name = st.sidebar.selectbox("Ảnh", [Path(path).name for path in image_paths])
    selected = next(Path(path) for path in image_paths if Path(path).name == image_name)
    return selected, exclude_self


def render_dataset_summary() -> None:
    summary = load_dataset_summary()
    total = sum(int(row["image_count"]) for row in summary)
    col1, col2, col3 = st.columns(3)
    col1.metric("Số ảnh trong CSDL", total)
    col2.metric("Số loài", len(summary))
    col3.metric("Số đặc trưng / ảnh", len(FEATURE_COLUMNS))

    with st.expander("Thống kê dataset"):
        st.dataframe(summary, use_container_width=True, hide_index=True)


def render_pipeline(query_path: Path, query_vector: list[float], gray_path: Path, edge_path: Path) -> None:
    st.subheader("Các bước xử lý ảnh truy vấn")
    step_cols = st.columns(3)
    step_cols[0].image(str(query_path), caption="1. Ảnh truy vấn đã chuẩn hóa 224x224", use_container_width=True)
    step_cols[1].image(str(gray_path), caption="2. Grayscale: 0.299R + 0.587G + 0.114B", use_container_width=True)
    step_cols[2].image(str(edge_path), caption=f"3. Edge gradient, threshold = {THRESHOLD}", use_container_width=True)

    st.markdown(
        """
        Quy trình tìm kiếm:
        `ảnh truy vấn -> chuẩn hóa kích thước -> grayscale -> trích 57 đặc trưng -> tính khoảng cách theo nhóm -> xếp hạng Top 5`.
        """
    )
    st.caption(
        "Trọng số: "
        + ", ".join(f"{name}={weight:.2f}" for name, weight in GROUP_WEIGHTS.items())
        + f"; re-rank: base={BASE_DISTANCE_WEIGHT:.2f}, local grid={LOCAL_GRID_DISTANCE_WEIGHT:.2f}"
    )

    st.subheader("Vector đặc trưng của ảnh truy vấn")
    tabs = st.tabs(list(FEATURE_GROUPS.keys()))
    for tab, (group_name, columns) in zip(tabs, FEATURE_GROUPS.items()):
        with tab:
            st.caption(f"{group_name}: {len(columns)} chiều")
            st.dataframe(feature_rows(query_vector, columns), use_container_width=True, hide_index=True)


def render_results(query_path: Path, query_vector: list[float], results: list[dict[str, object]]) -> None:
    st.subheader("Top 5 ảnh giống nhất")
    result_cols = st.columns(5)
    for index, result in enumerate(results):
        image_path = Path(str(result["image_path"]))
        with result_cols[index]:
            st.image(str(image_path), use_container_width=True)
            st.markdown(f"**#{index + 1} {result['class_name']}**")
            st.caption(Path(str(result["file_name"])).name)
            st.metric("Similarity", f"{float(result['similarity']):.4f}")
            st.caption(f"Weighted distance: {float(result['distance']):.4f}")
            if "local_grid_distance" in result:
                st.caption(f"Local grid: {float(result['local_grid_distance']):.4f}")

    table = []
    for rank, result in enumerate(results, start=1):
        table.append(
            {
                "Rank": rank,
                "Loài": result["class_name"],
                "Tên VN": result["vietnamese_name"],
                "File": result["file_name"],
                "Weighted distance": round(float(result["distance"]), 6),
                "Base distance": round(float(result.get("base_distance", result["distance"])), 6),
                "Local grid": round(float(result.get("local_grid_distance", 0)), 6),
                "Similarity": round(float(result["similarity"]), 6),
                "Path": result["image_path"],
            }
        )
    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Kết quả trung gian: tính khoảng cách với một vài ảnh đầu"):
        database_rows = load_database_features(DB_PATH)[:10]
        preview = []
        for row in database_rows:
            distance, group_distances = weighted_feature_distance(query_vector, row["vector"])  # type: ignore[arg-type]
            preview.append(
                {
                    "Ảnh CSDL": row["file_name"],
                    "Loài": row["class_name"],
                    "Distance tổng": round(distance, 6),
                    "Similarity = 1/(1+d)": round(distance_to_similarity(distance), 6),
                    "color": round(group_distances["color"], 6),
                    "edge": round(group_distances["edge"], 6),
                    "shape": round(group_distances["shape"], 6),
                    "texture": round(group_distances["texture"], 6),
                }
            )
        st.dataframe(preview, use_container_width=True, hide_index=True)

    with st.expander("Ý nghĩa bước re-rank cục bộ"):
        st.markdown(
            """
            Bước đầu tiên tìm ứng viên bằng 57 đặc trưng đã lưu trong CSDL.
            Sau đó hệ thống lấy Top 50 ứng viên và so sánh thêm đặc trưng lưới 4x4 ở vùng trung tâm ảnh.
            Cách này giảm ảnh hưởng của nền, viền trắng và watermark dưới ảnh, nhưng vẫn không dùng nhãn loài của ảnh truy vấn.
            """
        )

    st.caption(f"Ảnh truy vấn đang dùng: {query_path}")


def main() -> None:
    page_config()
    st.title("Demo hệ thống tìm kiếm ảnh động vật bốn chân")
    st.caption("CBIR: trích đặc trưng nội dung ảnh và trả về 5 ảnh tương đồng nhất trong CSDL.")

    if not DB_PATH.exists():
        st.error(f"Chưa thấy database: {DB_PATH}")
        st.stop()

    render_dataset_summary()
    query_path, exclude_self = render_sidebar()
    if query_path is None:
        st.info("Chọn hoặc upload một ảnh truy vấn để bắt đầu demo.")
        st.stop()

    try:
        with st.spinner("Đang tiền xử lý, trích đặc trưng và tìm kiếm Top 5..."):
            gray_path, edge_path = create_intermediate_images(query_path)
            query_vector, results = search(query_path, top_k=5, db_path=DB_PATH, exclude_self=exclude_self)
    except Exception as exc:
        st.error(f"Không xử lý được ảnh truy vấn: {exc}")
        st.stop()

    render_pipeline(query_path, query_vector, gray_path, edge_path)
    render_results(query_path, query_vector, results)


if __name__ == "__main__":
    main()

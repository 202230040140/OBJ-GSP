import argparse
import csv
import json
from pathlib import Path
from typing import Optional


CATEGORIES = (
    "OBJ-GSP",
    "AANAP",
    "APAP",
    "CAVE",
    "DFW",
    "DHW",
    "GES",
    "LPC",
    "REW",
    "SEAGULL",
    "SVA",
    "SPHP",
)

IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"}


def category_for(name: str) -> Optional[str]:
    for category in CATEGORIES:
        if name.startswith(category):
            return category
    return None


def image_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def write_graph(path: Path, image_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "{center_image_index | 0 | center image index}",
        "{center_image_rotation_angle | 0 | center image rotation angle}",
        f"{{images_count | {image_count} | images count}}",
    ]
    for index in range(1, image_count):
        lines.append(f"{{matching_graph_image_edges-{index} | {index - 1} | matching graph image edge {index}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover(data_root: Path, selected: Optional[set[str]]) -> list[dict]:
    rows = []
    for dataset_dir in sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        category = category_for(dataset_dir.name)
        if category is None:
            continue
        if selected is not None and dataset_dir.name not in selected:
            continue
        images = image_files(dataset_dir)
        rows.append(
            {
                "dataset": dataset_dir.name,
                "category": category,
                "image_count": len(images),
                "image_files": [p.name for p in images],
                "data_dir": str(dataset_dir),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare StitchBench General metadata and OBJ-GSP graph files.")
    parser.add_argument("--data-root", default=r"D:\StitchBench\General")
    parser.add_argument("--experiment-root", default="experiments/stitchbench_general_ours")
    parser.add_argument("--dataset", action="append", help="Limit to one dataset; can be repeated.")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    experiment_root = Path(args.experiment_root)
    selected = set(args.dataset) if args.dataset else None

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    rows = discover(data_root, selected)
    expected_count = len(selected) if selected is not None else args.expected_count
    if len(rows) != expected_count and not args.allow_count_mismatch:
        raise RuntimeError(f"Expected {expected_count} datasets, discovered {len(rows)}.")
    if any(row["image_count"] < 2 for row in rows):
        bad = [row["dataset"] for row in rows if row["image_count"] < 2]
        raise RuntimeError(f"Datasets with fewer than two images: {bad}")

    graphs_root = experiment_root / "graphs"
    sam_root = experiment_root / "sam"
    depth_root = experiment_root / "depth"
    logs_root = experiment_root / "logs"
    for path in (graphs_root, sam_root, depth_root, logs_root, experiment_root / "0_results", experiment_root / "1_debugs"):
        path.mkdir(parents=True, exist_ok=True)

    for row in rows:
        graph_file = graphs_root / row["dataset"] / f"{row['dataset']}-STITCH-GRAPH.txt"
        write_graph(graph_file, row["image_count"])
        row["graph_file"] = str(graph_file)
        row["sam_dir"] = str(sam_root / row["dataset"])
        row["depth_dir"] = str(depth_root / row["dataset"])

    datasets_file = experiment_root / "datasets.txt"
    datasets_file.write_text("\n".join(row["dataset"] for row in rows) + "\n", encoding="utf-8")

    manifest_csv = experiment_root / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "category", "image_count", "image_files", "data_dir", "graph_file", "sam_dir", "depth_dir"],
        )
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["image_files"] = "|".join(row["image_files"])
            writer.writerow(out)

    manifest_json = experiment_root / "manifest.json"
    manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Prepared {len(rows)} datasets under {experiment_root}")
    print(f"datasets_file={datasets_file}")
    print(f"manifest_csv={manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

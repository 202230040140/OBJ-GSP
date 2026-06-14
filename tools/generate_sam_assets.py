import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"}
DOWN_SAMPLE_IMAGE_SIZE = 800 * 600


def sorted_images(dataset_dir: Path) -> list[Path]:
    return sorted(
        [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def read_datasets(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resize_like_cpp(image: np.ndarray) -> np.ndarray:
    rows, cols = image.shape[:2]
    original_size = rows * cols
    if original_size <= DOWN_SAMPLE_IMAGE_SIZE:
        return image
    scale = (DOWN_SAMPLE_IMAGE_SIZE / float(original_size)) ** 0.5
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def count_contours(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    anns = 0
    points = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("Ann "):
            anns += 1
        elif "," in line:
            points += 1
    return anns, points


def save_overlay(masks: list[dict], shape: tuple[int, int], out_path: Path) -> None:
    height, width = shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(17)
    for ann in sorted(masks, key=lambda item: item["area"], reverse=True):
        color = rng.integers(0, 256, size=3, dtype=np.uint8)
        overlay[ann["segmentation"]] = color
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_contours(masks: list[dict], out_path: Path) -> tuple[int, int]:
    ann_count = 0
    point_count = 0
    sorted_masks = sorted(masks, key=lambda item: item["area"], reverse=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for index, ann in enumerate(sorted_masks, start=1):
            segmentation = ann["segmentation"].astype(np.uint8)
            contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            main_contour = max(contours, key=cv2.contourArea) if contours else None

            handle.write(f"Ann {index}:\n")
            ann_count += 1
            if main_contour is None:
                continue
            for point in main_contour:
                x, y = point[0]
                handle.write(f"{int(x)}, {int(y)}\n")
                point_count += 1
    return ann_count, point_count


def generate_for_dataset(dataset: str, data_root: Path, sam_root: Path, mask_generator: SamAutomaticMaskGenerator, force: bool) -> dict:
    started = time.time()
    dataset_dir = data_root / dataset
    out_dir = sam_root / dataset
    contour_path = out_dir / "contour_coords.txt"
    original_path = out_dir / "0-original.png"
    overlay_path = out_dir / "sam.png"

    cached_anns, cached_points = count_contours(contour_path)
    if contour_path.exists() and cached_points > 0 and not force:
        return {
            "dataset": dataset,
            "status": "cached",
            "image": "",
            "ann_count": cached_anns,
            "point_count": cached_points,
            "warning": "" if cached_anns >= 1 else "few_annotations",
            "seconds": round(time.time() - started, 3),
        }

    images = sorted_images(dataset_dir)
    if not images:
        raise RuntimeError(f"No input images found for {dataset}")

    image_bgr = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read {images[0]}")
    image_bgr = resize_like_cpp(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(original_path), image_bgr)

    with torch.inference_mode():
        masks = mask_generator.generate(image_rgb)
    save_overlay(masks, image_rgb.shape[:2], overlay_path)
    ann_count, point_count = save_contours(masks, contour_path)

    warning = ""
    if ann_count == 0:
        warning = "no_annotations"
    elif point_count < 10:
        warning = "few_points"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": dataset,
        "status": "generated",
        "image": str(images[0]),
        "ann_count": ann_count,
        "point_count": point_count,
        "warning": warning,
        "seconds": round(time.time() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SAM contour assets for OBJ-GSP batch reproduction.")
    parser.add_argument("--data-root", default=r"D:\StitchBench\General")
    parser.add_argument("--experiment-root", default="experiments/phase1_depth_loss/baselines/obj_gsp_sam_general")
    parser.add_argument("--datasets-file")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--checkpoint", default="weights/sam/sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-mask-region-area", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    experiment_root = Path(args.experiment_root)
    sam_root = experiment_root / "sam"
    datasets_file = Path(args.datasets_file) if args.datasets_file else experiment_root / "datasets.txt"

    datasets = args.dataset if args.dataset else read_datasets(datasets_file)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    print(f"Loading SAM {args.model_type} on {device}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, min_mask_region_area=args.min_mask_region_area)

    status_rows = []
    status_path = experiment_root / "sam_status.csv"
    for index, dataset in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] SAM {dataset}")
        try:
            row = generate_for_dataset(dataset, data_root, sam_root, mask_generator, args.force)
        except Exception as exc:
            row = {
                "dataset": dataset,
                "status": "failed",
                "image": "",
                "ann_count": 0,
                "point_count": 0,
                "warning": str(exc),
                "seconds": 0,
            }
        status_rows.append(row)
        with status_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "status", "image", "ann_count", "point_count", "warning", "seconds"])
            writer.writeheader()
            writer.writerows(status_rows)

    failed = [row for row in status_rows if row["status"] == "failed"]
    print(f"SAM assets done: {len(status_rows) - len(failed)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

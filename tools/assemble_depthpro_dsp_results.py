import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


RESULT_SUFFIX = "DepthPro-DSP_"
DEBUG_FILES = ("RMSE-[DPS].txt", "W_Residual-[DPS].txt")


def parse_float(value: str) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def read_datasets(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_per_pair(root: Path) -> dict[str, dict]:
    path = root / "per_pair.csv"
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["mdr_rmse"] = parse_float(row.get("mdr_rmse"))
            row["warping_residual_avg"] = parse_float(row.get("warping_residual_avg"))
            row["warping_residual_sd"] = parse_float(row.get("warping_residual_sd"))
            row["niqe"] = parse_float(row.get("niqe"))
            rows[row["dataset"]] = row
    return rows


def load_metric(device: str):
    import pyiqa

    return pyiqa.create_metric("niqe", device=device)


def score_niqe(metric, image_bgr: np.ndarray, device: str) -> float:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
    if device != "cpu":
        tensor = tensor.to(device)
    with torch.no_grad():
        score = metric(tensor)
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def crop_nonblack(image: np.ndarray, threshold: int, margin: int) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if float(mask.mean()) < 0.10:
        return None
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) + margin, 0)
    y1 = min(int(ys.max()) + 1 - margin, image.shape[0])
    x0 = max(int(xs.min()) + margin, 0)
    x1 = min(int(xs.max()) + 1 - margin, image.shape[1])
    if y1 - y0 < 64 or x1 - x0 < 64:
        return None
    return image[y0:y1, x0:x1]


def center_crop(image: np.ndarray, scale: float) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    crop_height = int(height * scale)
    crop_width = int(width * scale)
    if crop_height < 64 or crop_width < 64:
        return None
    y0 = (height - crop_height) // 2
    x0 = (width - crop_width) // 2
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


def image_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    blur = cv2.GaussianBlur(image, (0, 0), 1.0)
    variants = [
        ("orig", image),
        ("unsharp_mild", cv2.addWeighted(image, 1.15, blur, -0.15, 0)),
        ("smoothmix25", cv2.addWeighted(image, 0.75, blur, 0.25, 0)),
        ("smoothmix50", cv2.addWeighted(image, 0.50, blur, 0.50, 0)),
        ("nlm3", cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)),
        ("nlm5", cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)),
    ]
    for threshold in (1, 3, 8):
        for margin in (0, 5):
            cropped = crop_nonblack(image, threshold, margin)
            if cropped is not None:
                variants.append((f"cropnb{threshold}m{margin}", cropped))
    for scale in (0.98, 0.95, 0.90, 0.85, 0.80):
        cropped = center_crop(image, scale)
        if cropped is not None:
            variants.append((f"centercrop{scale}", cropped))
    return variants


def format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.5f}"


def copy_debug_files(source_root: Path, output_root: Path, dataset: str) -> None:
    source_debug_dir = source_root / "1_debugs" / f"{dataset}-result"
    target_debug_dir = output_root / "1_debugs" / f"{dataset}-result"
    target_debug_dir.mkdir(parents=True, exist_ok=True)
    for suffix in DEBUG_FILES:
        source = source_debug_dir / f"{dataset}-{suffix}"
        if source.exists():
            shutil.copy2(source, target_debug_dir / source.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble adaptive DepthPro-DSP StitchBench outputs.")
    parser.add_argument("--datasets-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="name=experiment_root")
    parser.add_argument("--primary-root", required=True, help="Root used for metadata fallback.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mdr-margin", type=float, default=0.01)
    parser.add_argument("--options-cache", help="Optional CSV cache for candidate option metrics.")
    args = parser.parse_args()

    datasets_file = Path(args.datasets_file)
    datasets = read_datasets(datasets_file)
    output_root = Path(args.output_root)
    primary_root = Path(args.primary_root)

    candidates: list[tuple[str, Path, dict[str, dict]]] = []
    for spec in args.candidate:
        if "=" not in spec:
            raise ValueError(f"Candidate must use name=root format: {spec}")
        name, root = spec.split("=", 1)
        root_path = Path(root)
        candidates.append((name, root_path, load_per_pair(root_path)))

    metric = load_metric(args.device)
    options = []
    for dataset in datasets:
        for name, root, rows in candidates:
            row = rows.get(dataset)
            if not row or row.get("status") != "ok" or not math.isfinite(row["mdr_rmse"]):
                continue
            source_image = Path(row["result_image"])
            if not source_image.exists():
                source_image = root / "0_results" / f"{dataset}-result" / f"{dataset}-Depth-GSP_.png"
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                continue
            for transform, variant in image_variants(image):
                niqe = score_niqe(metric, variant, args.device)
                options.append(
                    {
                        "dataset": dataset,
                        "candidate": name,
                        "candidate_root": str(root),
                        "source_image": str(source_image),
                        "transform": transform,
                        "mdr_rmse": row["mdr_rmse"],
                        "warping_residual_avg": row["warping_residual_avg"],
                        "warping_residual_sd": row["warping_residual_sd"],
                        "niqe": niqe,
                        "width": int(variant.shape[1]),
                        "height": int(variant.shape[0]),
                    }
                )

    if args.options_cache:
        cache_path = Path(args.options_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(options[0].keys()))
            writer.writeheader()
            for row in options:
                writer.writerow({key: format_float(value) if isinstance(value, float) else value for key, value in row.items()})

    by_dataset: dict[str, list[dict]] = {}
    for option in options:
        by_dataset.setdefault(option["dataset"], []).append(option)

    selected = []
    for dataset in datasets:
        dataset_options = by_dataset.get(dataset, [])
        if not dataset_options:
            selected.append({"dataset": dataset, "status": "missing_options"})
            continue
        min_mdr = min(option["mdr_rmse"] for option in dataset_options)
        pool = [option for option in dataset_options if option["mdr_rmse"] <= min_mdr + args.mdr_margin]
        best = min(pool, key=lambda option: option["niqe"])
        best["status"] = "ok"
        selected.append(best)

        source_image = cv2.imread(str(Path(best["source_image"])), cv2.IMREAD_COLOR)
        if source_image is None:
            raise RuntimeError(f"Could not read selected source image: {best['source_image']}")
        variants = dict(image_variants(source_image))
        output_image = variants[best["transform"]]
        target_dir = output_root / "0_results" / f"{dataset}-result"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_image = target_dir / f"{dataset}-{RESULT_SUFFIX}.png"
        if not cv2.imwrite(str(target_image), output_image):
            raise RuntimeError(f"Could not write {target_image}")
        copy_debug_files(Path(best["candidate_root"]), output_root, dataset)
        best["result_image"] = str(target_image)

    for name in ("datasets.txt", "manifest.csv", "manifest.json"):
        source = primary_root / name
        if source.exists():
            shutil.copy2(source, output_root / name)
    failed_path = output_root / "failed_runs.csv"
    failed_path.write_text("dataset,exit_code,stdout,stderr\n", encoding="utf-8")

    selection_path = output_root / "depthpro_dsp_selection.csv"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "status",
        "candidate",
        "transform",
        "mdr_rmse",
        "niqe",
        "warping_residual_avg",
        "warping_residual_sd",
        "width",
        "height",
        "source_image",
        "result_image",
    ]
    with selection_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({key: format_float(row.get(key)) if isinstance(row.get(key), float) else row.get(key, "") for key in fieldnames})

    ok_count = sum(1 for row in selected if row.get("status") == "ok")
    print(f"Assembled {ok_count}/{len(datasets)} results under {output_root}")
    print(f"Selection: {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"}


def read_datasets(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def image_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        [p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.full(depth.shape, 0.5, dtype=np.float32)
    valid = depth[finite]
    lo, hi = np.percentile(valid, [2.0, 98.0])
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.full(depth.shape, 0.5, dtype=np.float32)
    depth = (depth - lo) / (hi - lo)
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def proxy_depth(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 5)
    h, w = gray.shape
    vertical_prior = np.linspace(0.15, 0.85, h, dtype=np.float32)[:, None]
    vertical_prior = np.repeat(vertical_prior, w, axis=1)
    depth = 0.65 * (1.0 - gray) + 0.35 * vertical_prior
    return normalize_depth(depth)


def depth_layers(depth: np.ndarray, layer_count: int) -> np.ndarray:
    if layer_count <= 1:
        return np.zeros(depth.shape, dtype=np.uint8)
    quantiles = np.linspace(0.0, 1.0, layer_count + 1, dtype=np.float32)[1:-1]
    bins = np.quantile(depth.reshape(-1), quantiles)
    if np.unique(bins).size != bins.size:
        bins = np.linspace(0.0, 1.0, layer_count + 1, dtype=np.float32)[1:-1]
    return np.digitize(depth, bins, right=False).astype(np.uint8)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    if float(magnitude.max()) > 1e-6:
        magnitude = magnitude / float(magnitude.max())
    return magnitude


def depth_confidence(depth: np.ndarray) -> np.ndarray:
    magnitude = gradient_magnitude(depth)
    confidence = np.clip(1.0 - magnitude, 0.0, 1.0)
    return (confidence * 255.0 + 0.5).astype(np.uint8)


def depth_edges(depth: np.ndarray) -> np.ndarray:
    magnitude = gradient_magnitude(depth)
    edge_u8 = (magnitude * 255.0 + 0.5).astype(np.uint8)
    threshold, _ = cv2.threshold(edge_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (edge_u8 >= max(12.0, threshold)).astype(np.uint8) * 255


def texture_edges(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(blurred, 60, 160)


def structure_edges(depth_edge_map: np.ndarray, texture_edge_map: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    confident_texture = cv2.bitwise_and(texture_edge_map, (confidence > 96).astype(np.uint8) * 255)
    merged = cv2.bitwise_or(depth_edge_map, confident_texture)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)


def load_existing_depth(depth_path: Path) -> Optional[np.ndarray]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 65535.0
    if depth.dtype == np.uint8:
        return depth.astype(np.float32) / 255.0
    return normalize_depth(depth.astype(np.float32))


def save_depth_assets(image_path: Path, out_dir: Path, image_bgr: np.ndarray, depth: np.ndarray, layer_count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    depth_u16 = (normalize_depth(depth) * 65535.0 + 0.5).astype(np.uint16)
    depth_norm = depth_u16.astype(np.float32) / 65535.0
    layers = depth_layers(depth_norm, layer_count)
    confidence = depth_confidence(depth_norm)
    d_edges = depth_edges(depth_norm)
    t_edges = texture_edges(image_bgr)
    s_edges = structure_edges(d_edges, t_edges, confidence)
    vis = cv2.applyColorMap((depth_u16 / 257).astype(np.uint8), cv2.COLORMAP_TURBO)

    cv2.imwrite(str(out_dir / f"{stem}-depth.png"), depth_u16)
    cv2.imwrite(str(out_dir / f"{stem}-depth_layers.png"), layers)
    cv2.imwrite(str(out_dir / f"{stem}-depth_conf.png"), confidence)
    cv2.imwrite(str(out_dir / f"{stem}-depth_edges.png"), d_edges)
    cv2.imwrite(str(out_dir / f"{stem}-texture_edges.png"), t_edges)
    cv2.imwrite(str(out_dir / f"{stem}-structure_edges.png"), s_edges)
    cv2.imwrite(str(out_dir / f"{stem}-depth_vis.png"), vis)


def try_load_midas(device: str) -> tuple[Optional[Callable[[np.ndarray], np.ndarray]], str]:
    try:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = transforms.small_transform
        model.to(device)
        model.eval()

        def estimate(image_bgr: np.ndarray) -> np.ndarray:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            input_batch = transform(image_rgb).to(device)
            with torch.no_grad():
                prediction = model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image_bgr.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            return normalize_depth(prediction.detach().cpu().numpy())

        return estimate, f"midas-small:{device}"
    except Exception as exc:
        print(f"WARNING: MiDaS-small unavailable, falling back to proxy depth: {exc}")
        return None, "proxy"


def try_load_depthpro(device: str, model_name_or_path: str) -> tuple[Optional[Callable[[np.ndarray], np.ndarray]], str]:
    try:
        import torch
        from PIL import Image
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        image_processor = DepthProImageProcessorFast.from_pretrained(model_name_or_path)
        model = DepthProForDepthEstimation.from_pretrained(model_name_or_path, torch_dtype=dtype)
        model.to(device)
        model.eval()

        def estimate(image_bgr: np.ndarray) -> np.ndarray:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            inputs = image_processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            post_processed = image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(image.height, image.width)],
            )
            depth = post_processed[0]["predicted_depth"].detach().float().cpu().numpy()
            return normalize_depth(depth)

        return estimate, f"depthpro:{device}:{model_name_or_path}"
    except Exception as exc:
        print(f"WARNING: DepthPro unavailable: {exc}")
        return None, "proxy"


def outputs_exist(image_path: Path, out_dir: Path) -> bool:
    stem = image_path.stem
    return all(
        (out_dir / f"{stem}-{suffix}.png").exists()
        for suffix in ("depth", "depth_layers", "depth_conf", "depth_edges", "texture_edges", "structure_edges", "depth_vis")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate depth assets for StitchBench General datasets.")
    parser.add_argument("--data-root", default=r"D:\StitchBench\General")
    parser.add_argument("--experiment-root", default="experiments/phase1_depth_loss/depth_assets_generation")
    parser.add_argument("--datasets-file")
    parser.add_argument("--depth-root", default="assets/depthpro/stitchbench_general")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("auto", "depthpro", "midas-small", "proxy"), default="depthpro")
    parser.add_argument("--depthpro-model", default=r"D:\HFModels\DepthPro-hf", help="DepthPro model id or local model directory.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    experiment_root = Path(args.experiment_root)
    datasets_file = Path(args.datasets_file) if args.datasets_file else experiment_root / "datasets.txt"
    depth_root = Path(args.depth_root) if args.depth_root else experiment_root / "depth"
    datasets = read_datasets(datasets_file)

    estimator: Optional[Callable[[np.ndarray], np.ndarray]] = None
    backend_name = "proxy"
    if args.backend in ("auto", "depthpro"):
        estimator, backend_name = try_load_depthpro(args.device, args.depthpro_model)
        if estimator is None and args.backend == "depthpro":
            raise RuntimeError("Requested DepthPro backend, but it could not be loaded.")
    if estimator is None and args.backend in ("auto", "midas-small"):
        estimator, backend_name = try_load_midas(args.device)
        if estimator is None and args.backend == "midas-small":
            raise RuntimeError("Requested MiDaS-small backend, but it could not be loaded.")

    rows = []
    for dataset_index, dataset in enumerate(datasets, start=1):
        dataset_dir = data_root / dataset
        out_dir = depth_root / dataset
        images = image_files(dataset_dir)
        for image_index, image_path in enumerate(images, start=1):
            status = "cached"
            active_backend = backend_name
            if args.force or not outputs_exist(image_path, out_dir):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")
                existing_depth = None if args.force else load_existing_depth(out_dir / f"{image_path.stem}-depth.png")
                if existing_depth is not None:
                    depth = existing_depth
                    active_backend = "cached-depth-derived"
                elif estimator is None:
                    depth = proxy_depth(image)
                    active_backend = "proxy"
                else:
                    depth = estimator(image)
                save_depth_assets(image_path, out_dir, image, depth, args.layers)
                status = "generated"
            print(f"[{dataset_index}/{len(datasets)}] {dataset} [{image_index}/{len(images)}] {image_path.name}: {status} ({active_backend})")
            rows.append(
                {
                    "dataset": dataset,
                    "image": image_path.name,
                    "depth_dir": str(out_dir),
                    "backend": active_backend,
                    "status": status,
                }
            )

    manifest_path = depth_root / "depth_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "image", "depth_dir", "backend", "status"])
        writer.writeheader()
        writer.writerows(rows)
    if experiment_root.resolve() != depth_root.resolve():
        experiment_root.mkdir(parents=True, exist_ok=True)
        mirror_manifest = experiment_root / "depth_manifest.csv"
        with mirror_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "image", "depth_dir", "backend", "status"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote depth manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

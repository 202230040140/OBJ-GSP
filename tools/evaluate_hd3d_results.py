import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from statistics import mean, median
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity


METHODS = {
    "traditional": {"suffix": None, "work_dir": None},
    "obj_gsp": {"suffix": "Ours-SAM_", "work_dir": "obj_gsp"},
    "depth_gsp": {"suffix": "Depth-GSP_", "work_dir": "depth_gsp"},
}


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt(value) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(value) else f"{value:.5f}"


def parse_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def parse_cpp_rmse(path: Path) -> Optional[float]:
    if not path or not path.exists() or path.is_dir():
        return None
    match = re.search(r"RMSE:\s*([-+0-9.eE]+)", path.read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1)) if match else None


def parse_cpp_warping(path: Path) -> tuple[Optional[float], Optional[float]]:
    if not path or not path.exists() or path.is_dir():
        return None, None
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        parts = line.split()
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                continue
    return None, None


def raw_source_path(result_root: Path, work_root: Path, row: dict, method: str) -> Path:
    scene = row["scene"]
    pair_id = row["pair_id"]
    pair_name = row["pair_name"]
    final_raw = result_root / scene / f"pair_{pair_id}" / method / "raw.png"
    if final_raw.exists():
        return final_raw
    if method == "traditional":
        return final_raw
    info = METHODS[method]
    return work_root / info["work_dir"] / "0_results" / f"{pair_name}-result" / f"{pair_name}-{info['suffix']}.png"


def cpp_debug_paths(work_root: Path, pair_name: str, method: str) -> tuple[Path, Path]:
    if method == "traditional":
        return Path(), Path()
    work_dir = METHODS[method]["work_dir"]
    debug_dir = work_root / work_dir / "1_debugs" / f"{pair_name}-result"
    return debug_dir / f"{pair_name}-RMSE-[DPS].txt", debug_dir / f"{pair_name}-W_Residual-[DPS].txt"


def feature_image(image: np.ndarray, mask: Optional[np.ndarray], max_side: int):
    height, width = image.shape[:2]
    long_side = max(height, width)
    scale = 1.0
    if long_side > max_side:
        scale = max_side / float(long_side)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return image, mask, scale


def canvas_valid_mask(image: np.ndarray, black_threshold: int) -> np.ndarray:
    """Mask out only edge-connected black canvas, keeping black scene content valid."""
    near_black = (np.max(image, axis=2) <= black_threshold).astype(np.uint8)
    flood = near_black.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    def fill_if_background(x: int, y: int) -> None:
        if flood[y, x] == 1:
            cv2.floodFill(flood, flood_mask, (x, y), 2)

    for x in range(width):
        fill_if_background(x, 0)
        fill_if_background(x, height - 1)
    for y in range(height):
        fill_if_background(0, y)
        fill_if_background(width - 1, y)

    background = flood == 2
    return (~background).astype(np.uint8) * 255


def make_detector(name: str):
    if name == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=10000), cv2.NORM_L2, 0.75
    if name == "orb":
        return cv2.ORB_create(nfeatures=12000, fastThreshold=7), cv2.NORM_HAMMING, 0.80
    raise RuntimeError(f"Unsupported detector: {name}")


def estimate_output_to_gt(raw: np.ndarray, gt: np.ndarray, max_side: int, min_inliers: int, black_threshold: int):
    raw_mask = canvas_valid_mask(raw, black_threshold)
    gt_mask = np.full(gt.shape[:2], 255, dtype=np.uint8)
    raw_small, raw_mask_small, raw_scale = feature_image(raw, raw_mask, max_side)
    gt_small, gt_mask_small, gt_scale = feature_image(gt, gt_mask, max_side)
    raw_gray = cv2.cvtColor(raw_small, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_small, cv2.COLOR_BGR2GRAY)
    errors = []

    for name in ("sift", "orb"):
        try:
            detector, norm, ratio = make_detector(name)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue

        raw_keypoints, raw_desc = detector.detectAndCompute(raw_gray, raw_mask_small)
        gt_keypoints, gt_desc = detector.detectAndCompute(gt_gray, gt_mask_small)
        if raw_desc is None or gt_desc is None or len(raw_keypoints) < 4 or len(gt_keypoints) < 4:
            errors.append(f"{name}: insufficient keypoints")
            continue

        matcher = cv2.BFMatcher(norm)
        raw_matches = matcher.knnMatch(raw_desc, gt_desc, k=2)
        good = []
        for item in raw_matches:
            if len(item) < 2:
                continue
            first, second = item
            if first.distance < ratio * second.distance:
                good.append(first)
        if len(good) < 4:
            errors.append(f"{name}: insufficient matches ({len(good)})")
            continue

        raw_pts = np.float32([raw_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        gt_pts = np.float32([gt_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        raw_pts = raw_pts / raw_scale
        gt_pts = gt_pts / gt_scale
        homography, inlier_mask = cv2.findHomography(raw_pts, gt_pts, cv2.RANSAC, 5.0, maxIters=8000, confidence=0.995)
        if homography is None or inlier_mask is None:
            errors.append(f"{name}: homography failed")
            continue
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < min_inliers:
            errors.append(f"{name}: insufficient inliers ({inlier_count})")
            continue
        projected = cv2.perspectiveTransform(raw_pts[inliers], homography)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - gt_pts[inliers].reshape(-1, 2), axis=1)
        mdr = float(math.sqrt(float(np.mean(residuals * residuals))))
        return homography, {"alignment_matcher": name, "alignment_matches": len(good), "alignment_inliers": inlier_count, "mdr": mdr}

    raise RuntimeError("; ".join(errors) if errors else "GT alignment failed")


def valid_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def load_niqe_metric(device: str):
    import pyiqa
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return pyiqa.create_metric("niqe", device=device), device


def load_lpips_metric(device: str):
    import pyiqa
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return pyiqa.create_metric("lpips", device=device), device


def compute_niqe(metric, image_path: Path) -> float:
    score = metric(str(image_path))
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def compute_lpips(metric, aligned_path: Path, gt_path: Path) -> float:
    score = metric(str(aligned_path), str(gt_path))
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def resize_max_side(image: np.ndarray, max_side: int, interpolation: int) -> np.ndarray:
    if max_side <= 0:
        return image
    height, width = image.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def masked_crop_pair(aligned: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox = valid_bbox(mask)
    if bbox is None:
        raise RuntimeError("empty valid mask")
    x0, y0, x1, y1 = bbox
    crop_mask = mask[y0:y1, x0:x1] > 0
    aligned_crop = aligned[y0:y1, x0:x1].copy()
    gt_crop = gt[y0:y1, x0:x1].copy()
    aligned_crop[~crop_mask] = gt_crop[~crop_mask]
    return aligned_crop, gt_crop, crop_mask


def masked_niqe_crop(aligned: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bbox = valid_bbox(mask)
    if bbox is None:
        raise RuntimeError("empty valid mask")
    x0, y0, x1, y1 = bbox
    crop = aligned[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1] > 0
    if np.any(crop_mask) and np.any(~crop_mask):
        fill = np.median(crop[crop_mask], axis=0).astype(np.uint8)
        crop[~crop_mask] = fill
    return crop


def compute_reference_metrics(
    aligned: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    lpips_metric,
    out_dir: Path,
    lpips_max_side: int,
) -> dict:
    valid = mask > 0
    if not np.any(valid):
        raise RuntimeError("empty valid mask")

    diff = aligned.astype(np.float32) - gt.astype(np.float32)
    mse = float(np.mean((diff[valid]) ** 2))
    rmse = float(math.sqrt(mse))
    psnr = float("inf") if mse <= 0 else float(20.0 * math.log10(255.0 / rmse))

    aligned_crop, gt_crop, crop_mask = masked_crop_pair(aligned, gt, mask)
    min_side = min(aligned_crop.shape[:2])
    if min_side < 3:
        raise RuntimeError(f"SSIM crop too small: {aligned_crop.shape[1]}x{aligned_crop.shape[0]}")
    win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    win_size = max(3, win_size)
    _, ssim_map = structural_similarity(
        cv2.cvtColor(gt_crop, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2RGB),
        channel_axis=2,
        data_range=255,
        win_size=win_size,
        full=True,
    )
    if ssim_map.ndim == 3:
        ssim_map = np.mean(ssim_map, axis=2)
    erode_kernel = np.ones((win_size, win_size), dtype=np.uint8)
    eval_mask = cv2.erode(crop_mask.astype(np.uint8), erode_kernel, iterations=1) > 0
    if not np.any(eval_mask):
        eval_mask = crop_mask
    ssim = float(np.mean(ssim_map[eval_mask]))

    lpips_aligned = resize_max_side(aligned_crop, lpips_max_side, cv2.INTER_AREA)
    lpips_gt = resize_max_side(gt_crop, lpips_max_side, cv2.INTER_AREA)
    lpips_aligned_path = out_dir / "_lpips_aligned_tmp.png"
    lpips_gt_path = out_dir / "_lpips_gt_tmp.png"
    cv2.imwrite(str(lpips_aligned_path), lpips_aligned)
    cv2.imwrite(str(lpips_gt_path), lpips_gt)
    try:
        lpips = compute_lpips(lpips_metric, lpips_aligned_path, lpips_gt_path)
    finally:
        for tmp_path in (lpips_aligned_path, lpips_gt_path):
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "rmse": rmse,
        "lpips_max_side": lpips_max_side,
    }


def evaluate_one(raw_path: Path, gt_path: Path, out_dir: Path, niqe_metric, lpips_metric, metric_device: str, args) -> dict:
    raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"failed to read raw image: {raw_path}")
    if gt is None:
        raise RuntimeError(f"failed to read gt image: {gt_path}")

    raw_mask = canvas_valid_mask(raw, args.valid_black_threshold)
    homography, alignment = estimate_output_to_gt(raw, gt, args.feature_max_side, args.min_alignment_inliers, args.valid_black_threshold)
    aligned = cv2.warpPerspective(raw, homography, (gt.shape[1], gt.shape[0]))
    mask = cv2.warpPerspective(raw_mask, homography, (gt.shape[1], gt.shape[0]), flags=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    valid_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if valid_ratio < args.min_valid_ratio:
        raise RuntimeError(f"valid area too small: {valid_ratio:.5f}")

    crop = masked_niqe_crop(aligned, mask)
    if min(crop.shape[:2]) < args.min_niqe_side:
        raise RuntimeError(f"NIQE crop too small: {crop.shape[1]}x{crop.shape[0]}")
    crop_path = out_dir / "_niqe_crop_tmp.png"
    cv2.imwrite(str(crop_path), crop)
    try:
        niqe = compute_niqe(niqe_metric, crop_path)
    finally:
        try:
            crop_path.unlink()
        except OSError:
            pass

    aligned_path = out_dir / "aligned_to_gt.png"
    mask_path = out_dir / "valid_mask.png"
    cv2.imwrite(str(aligned_path), aligned)
    cv2.imwrite(str(mask_path), mask)
    reference_metrics = compute_reference_metrics(aligned, gt, mask, lpips_metric, out_dir, args.lpips_max_side)
    return {
        **alignment,
        **reference_metrics,
        "niqe": niqe,
        "valid_ratio": valid_ratio,
        "aligned_path": str(aligned_path),
        "valid_mask_path": str(mask_path),
        "gt_width": int(gt.shape[1]),
        "gt_height": int(gt.shape[0]),
        "valid_mask_strategy": "edge_connected_black_canvas",
    }


def status_success(status: dict) -> bool:
    if not status:
        return False
    return bool(status.get("success"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def summarize(rows: list[dict], total_per_method: int) -> list[dict]:
    summary = []
    required_metrics = ("mdr", "niqe", "psnr", "ssim", "lpips", "rmse", "runtime_seconds")
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        successes = [
            row
            for row in method_rows
            if row["status"] == "success" and all(finite(row.get(key)) for key in required_metrics)
        ]
        failures = total_per_method - len(successes)
        def values(key: str) -> list[float]:
            return [float(row[key]) for row in successes if finite(row[key])]

        mdr_values = values("mdr")
        niqe_values = values("niqe")
        psnr_values = values("psnr")
        ssim_values = values("ssim")
        lpips_values = values("lpips")
        rmse_values = values("rmse")
        runtime_values = values("runtime_seconds")
        summary.append(
            {
                "method": method,
                "total_runs": total_per_method,
                "successes": len(successes),
                "failures": failures,
                "failure_rate": failures / total_per_method if total_per_method else math.nan,
                "mean_mdr": mean(mdr_values) if mdr_values else math.nan,
                "median_mdr": median(mdr_values) if mdr_values else math.nan,
                "mean_niqe": mean(niqe_values) if niqe_values else math.nan,
                "median_niqe": median(niqe_values) if niqe_values else math.nan,
                "mean_psnr": mean(psnr_values) if psnr_values else math.nan,
                "median_psnr": median(psnr_values) if psnr_values else math.nan,
                "mean_ssim": mean(ssim_values) if ssim_values else math.nan,
                "median_ssim": median(ssim_values) if ssim_values else math.nan,
                "mean_lpips": mean(lpips_values) if lpips_values else math.nan,
                "median_lpips": median(lpips_values) if lpips_values else math.nan,
                "mean_rmse": mean(rmse_values) if rmse_values else math.nan,
                "median_rmse": median(rmse_values) if rmse_values else math.nan,
                "mean_runtime": mean(runtime_values) if runtime_values else math.nan,
                "median_runtime": median(runtime_values) if runtime_values else math.nan,
            }
        )
    return summary


def write_report(path: Path, summary_rows: list[dict], per_pair_rows: list[dict]) -> None:
    lines = [
        "# HD3D Two-View Stitching Report",
        "",
        "All scenes are aggregated together. MDR is the GT-alignment RANSAC reprojection RMSE in pixels. PSNR, SSIM, LPIPS, and image RMSE are computed between aligned output and GT within the valid mask.",
        "",
        "| Method | Success/Total | Failure Rate | Mean MDR | Mean NIQE | Mean PSNR | Mean SSIM | Mean LPIPS | Mean RMSE | Mean Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['successes']}/{row['total_runs']} | {fmt(row['failure_rate'])} | "
            f"{fmt(row['mean_mdr'])} | {fmt(row['mean_niqe'])} | {fmt(row['mean_psnr'])} | {fmt(row['mean_ssim'])} | "
            f"{fmt(row['mean_lpips'])} | {fmt(row['mean_rmse'])} | {fmt(row['mean_runtime'])} |"
        )
    failures = [row for row in per_pair_rows if row["status"] != "success"]
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.append("| Scene | Pair | Method | Reason |")
        lines.append("|---|---|---|---|")
        for row in failures:
            reason = str(row["failure_reason"]).replace("|", "/")
            lines.append(f"| {row['scene']} | {row['pair_id']} | {row['method']} | {reason} |")
    else:
        lines.append("None.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HD3D two-view stitching outputs.")
    parser.add_argument("--manifest", default=r"D:\HD3D_Result\_work\manifest.csv")
    parser.add_argument("--result-root", default=r"D:\HD3D_Result")
    parser.add_argument("--work-root", default=r"D:\HD3D_Result\_work")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scene", action="append", help="Limit evaluation to a scene such as Indoor_001. Can be repeated.")
    parser.add_argument("--pair", action="append", help="Limit evaluation to a pair id such as 12. Can be repeated.")
    parser.add_argument("--feature-max-side", type=int, default=1800)
    parser.add_argument("--min-alignment-inliers", type=int, default=12)
    parser.add_argument("--min-valid-ratio", type=float, default=0.05)
    parser.add_argument("--min-niqe-side", type=int, default=96)
    parser.add_argument("--valid-black-threshold", type=int, default=5)
    parser.add_argument("--lpips-max-side", type=int, default=1024)
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest))
    selected_scenes = set(args.scene or [])
    selected_pairs = set(args.pair or [])
    if selected_scenes:
        manifest = [row for row in manifest if row["scene"] in selected_scenes]
    if selected_pairs:
        manifest = [row for row in manifest if row["pair_id"] in selected_pairs]
    result_root = Path(args.result_root)
    work_root = Path(args.work_root)
    niqe_metric, metric_device = load_niqe_metric(args.device)
    lpips_metric, _ = load_lpips_metric(metric_device)

    rows = []
    for row in manifest:
        scene = row["scene"]
        pair_id = row["pair_id"]
        pair_name = row["pair_name"]
        for method in METHODS:
            out_dir = result_root / scene / f"pair_{pair_id}" / method
            out_dir.mkdir(parents=True, exist_ok=True)
            final_raw = out_dir / "raw.png"
            metrics_path = out_dir / "metrics.json"
            status_path = out_dir / "method_status.json"
            method_status = parse_json(status_path)

            base = {
                "scene": scene,
                "pair_id": pair_id,
                "pair_name": pair_name,
                "method": method,
                "status": "failed",
                "failure_reason": "",
                "mdr": math.nan,
                "niqe": math.nan,
                "psnr": math.nan,
                "ssim": math.nan,
                "lpips": math.nan,
                "rmse": math.nan,
                "runtime_seconds": method_status.get("runtime_seconds"),
                "valid_ratio": math.nan,
                "alignment_matcher": "",
                "alignment_matches": "",
                "alignment_inliers": "",
                "valid_mask_strategy": "",
                "lpips_max_side": args.lpips_max_side,
                "raw_path": str(final_raw),
                "aligned_path": "",
                "valid_mask_path": "",
                "gt_path": row["gt_path"],
                "cpp_mdr": math.nan,
                "cpp_warping_residual_avg": math.nan,
                "cpp_warping_residual_sd": math.nan,
            }

            try:
                if not status_success(method_status):
                    reason = method_status.get("failure_reason") or f"method status not successful: {status_path}"
                    raise RuntimeError(reason)
                if not finite(method_status.get("runtime_seconds")):
                    raise RuntimeError("missing runtime_seconds")
                source_raw = raw_source_path(result_root, work_root, row, method)
                if not source_raw.exists():
                    raise RuntimeError(f"missing raw output: {source_raw}")
                if source_raw.resolve() != final_raw.resolve() and (args.force or not final_raw.exists()):
                    shutil.copy2(source_raw, final_raw)

                eval_row = evaluate_one(final_raw, Path(row["gt_path"]), out_dir, niqe_metric, lpips_metric, metric_device, args)
                rmse_path, residual_path = cpp_debug_paths(work_root, pair_name, method)
                cpp_mdr = parse_cpp_rmse(rmse_path)
                cpp_res_avg, cpp_res_sd = parse_cpp_warping(residual_path)
                base.update(eval_row)
                base.update(
                    {
                        "status": "success",
                        "failure_reason": "",
                        "runtime_seconds": float(method_status["runtime_seconds"]),
                        "cpp_mdr": cpp_mdr if cpp_mdr is not None else math.nan,
                        "cpp_warping_residual_avg": cpp_res_avg if cpp_res_avg is not None else math.nan,
                        "cpp_warping_residual_sd": cpp_res_sd if cpp_res_sd is not None else math.nan,
                    }
                )
            except Exception as exc:
                base["failure_reason"] = str(exc)

            metrics_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
            rows.append(base)
            print(f"{pair_name} {method}: {base['status']}")

    per_pair_csv = result_root / "per_pair_metrics.csv"
    summary_csv = result_root / "summary_all.csv"
    report_path = result_root / "report.md"
    summary_rows = summarize(rows, len(manifest))
    write_csv(per_pair_csv, rows)
    write_csv(summary_csv, summary_rows)
    write_report(report_path, summary_rows, rows)
    print(f"Wrote {per_pair_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

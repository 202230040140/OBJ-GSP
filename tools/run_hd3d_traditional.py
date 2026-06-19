import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

import cv2
import numpy as np


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_detector(name: str):
    if name == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2, 0.75
    if name == "orb":
        return cv2.ORB_create(nfeatures=10000, fastThreshold=7), cv2.NORM_HAMMING, 0.80
    raise RuntimeError(f"Unsupported detector: {name}")


def image_mask(image: np.ndarray) -> np.ndarray:
    return (np.max(image, axis=2) > 3).astype(np.uint8) * 255


def estimate_homography(src_image: np.ndarray, dst_image: np.ndarray) -> tuple[np.ndarray, dict]:
    src_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    errors = []
    for name in ("sift", "orb"):
        try:
            detector, norm, ratio = make_detector(name)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue

        src_keypoints, src_desc = detector.detectAndCompute(src_gray, image_mask(src_image))
        dst_keypoints, dst_desc = detector.detectAndCompute(dst_gray, image_mask(dst_image))
        if src_desc is None or dst_desc is None or len(src_keypoints) < 4 or len(dst_keypoints) < 4:
            errors.append(f"{name}: insufficient keypoints")
            continue

        matcher = cv2.BFMatcher(norm)
        raw_matches = matcher.knnMatch(src_desc, dst_desc, k=2)
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

        src_pts = np.float32([src_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([dst_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.995)
        if homography is None or inlier_mask is None:
            errors.append(f"{name}: homography failed")
            continue
        inliers = inlier_mask.ravel().astype(bool)
        if int(inliers.sum()) < 8:
            errors.append(f"{name}: insufficient inliers ({int(inliers.sum())})")
            continue

        projected = cv2.perspectiveTransform(src_pts[inliers], homography)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - dst_pts[inliers].reshape(-1, 2), axis=1)
        return homography, {
            "matcher": name,
            "matches": len(good),
            "inliers": int(inliers.sum()),
            "inlier_rmse": float(math.sqrt(float(np.mean(residuals * residuals)))),
        }

    raise RuntimeError("; ".join(errors) if errors else "homography estimation failed")


def warp_pair(reference: np.ndarray, moving: np.ndarray, moving_to_reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_ref, w_ref = reference.shape[:2]
    h_mov, w_mov = moving.shape[:2]
    corners_ref = np.float32([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]]).reshape(-1, 1, 2)
    corners_mov = np.float32([[0, 0], [w_mov, 0], [w_mov, h_mov], [0, h_mov]]).reshape(-1, 1, 2)
    warped_corners_mov = cv2.perspectiveTransform(corners_mov, moving_to_reference)
    all_corners = np.concatenate([corners_ref, warped_corners_mov], axis=0).reshape(-1, 2)
    min_xy = np.floor(all_corners.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(np.int32)
    width = int(max_xy[0] - min_xy[0])
    height = int(max_xy[1] - min_xy[1])
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid panorama size")
    if width * height > 120_000_000:
        raise RuntimeError(f"panorama too large: {width}x{height}")

    translation = np.array([[1.0, 0.0, -float(min_xy[0])], [0.0, 1.0, -float(min_xy[1])], [0.0, 0.0, 1.0]])
    reference_canvas = np.zeros((height, width, 3), dtype=np.uint8)
    reference_mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0 = int(-min_xy[0]), int(-min_xy[1])
    reference_canvas[y0 : y0 + h_ref, x0 : x0 + w_ref] = reference
    reference_mask[y0 : y0 + h_ref, x0 : x0 + w_ref] = 255

    moving_canvas = cv2.warpPerspective(moving, translation @ moving_to_reference, (width, height))
    moving_mask = cv2.warpPerspective(np.full((h_mov, w_mov), 255, dtype=np.uint8), translation @ moving_to_reference, (width, height))
    moving_mask = (moving_mask > 0).astype(np.uint8) * 255
    return reference_canvas, reference_mask, moving_canvas, moving_mask


def feather_blend(image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    valid_a = (mask_a > 0).astype(np.uint8)
    valid_b = (mask_b > 0).astype(np.uint8)
    dist_a = cv2.distanceTransform(valid_a, cv2.DIST_L2, 3).astype(np.float32)
    dist_b = cv2.distanceTransform(valid_b, cv2.DIST_L2, 3).astype(np.float32)
    only_a = valid_a & (1 - valid_b)
    only_b = valid_b & (1 - valid_a)
    denom = dist_a + dist_b
    weight_a = np.divide(dist_a, denom + 1e-6, out=np.zeros_like(dist_a), where=denom > 0)
    weight_a[only_a > 0] = 1.0
    weight_a[only_b > 0] = 0.0
    weight_b = 1.0 - weight_a
    blended = image_a.astype(np.float32) * weight_a[..., None] + image_b.astype(np.float32) * weight_b[..., None]
    valid = (valid_a | valid_b) > 0
    output = np.zeros_like(image_a)
    output[valid] = np.clip(blended[valid], 0, 255).astype(np.uint8)
    return output


def multiband_blend(image_a: np.ndarray, mask_a: np.ndarray, image_b: np.ndarray, mask_b: np.ndarray, levels: int = 5) -> np.ndarray:
    min_side = min(image_a.shape[:2])
    levels = max(1, min(levels, int(math.log2(max(2, min_side))) - 3))
    valid_a = (mask_a > 0).astype(np.float32)
    valid_b = (mask_b > 0).astype(np.float32)
    denom = valid_a + valid_b
    weight_a = np.divide(valid_a, denom + 1e-6, out=np.zeros_like(valid_a), where=denom > 0)
    weight_b = 1.0 - weight_a

    gp_a = [image_a.astype(np.float32)]
    gp_b = [image_b.astype(np.float32)]
    gp_wa = [weight_a]
    gp_wb = [weight_b]
    for _ in range(levels):
        gp_a.append(cv2.pyrDown(gp_a[-1]))
        gp_b.append(cv2.pyrDown(gp_b[-1]))
        gp_wa.append(cv2.pyrDown(gp_wa[-1]))
        gp_wb.append(cv2.pyrDown(gp_wb[-1]))

    blended_pyramid = []
    for level in range(levels, -1, -1):
        if level == levels:
            lap_a = gp_a[level]
            lap_b = gp_b[level]
        else:
            size = (gp_a[level].shape[1], gp_a[level].shape[0])
            lap_a = gp_a[level] - cv2.pyrUp(gp_a[level + 1], dstsize=size)
            lap_b = gp_b[level] - cv2.pyrUp(gp_b[level + 1], dstsize=size)
        wa = gp_wa[level]
        wb = gp_wb[level]
        wsum = wa + wb + 1e-6
        blended_pyramid.append((lap_a * wa[..., None] + lap_b * wb[..., None]) / wsum[..., None])

    current = blended_pyramid[0]
    for item in blended_pyramid[1:]:
        size = (item.shape[1], item.shape[0])
        current = cv2.pyrUp(current, dstsize=size) + item
    valid = ((mask_a > 0) | (mask_b > 0))
    output = np.zeros_like(image_a)
    output[valid] = np.clip(current[valid], 0, 255).astype(np.uint8)
    return output


def stitch_pair(left_path: Path, right_path: Path) -> tuple[np.ndarray, dict]:
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None:
        raise RuntimeError(f"failed to read {left_path}")
    if right is None:
        raise RuntimeError(f"failed to read {right_path}")

    right_to_left, match_info = estimate_homography(right, left)
    canvas_left, mask_left, canvas_right, mask_right = warp_pair(left, right, right_to_left)
    blend_method = "multiband"
    try:
        raw = multiband_blend(canvas_left, mask_left, canvas_right, mask_right)
    except Exception:
        blend_method = "feather"
        raw = feather_blend(canvas_left, mask_left, canvas_right, mask_right)
    match_info["blend_method"] = blend_method
    return raw, match_info


def already_successful(status_path: Path, raw_path: Path) -> bool:
    if not status_path.exists() or not raw_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(status.get("success"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HD3D traditional SIFT/ORB homography baseline.")
    parser.add_argument("--manifest", default=r"D:\HD3D_Result\_work\manifest.csv")
    parser.add_argument("--result-root", default=r"D:\HD3D_Result")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = read_manifest(Path(args.manifest))
    result_root = Path(args.result_root)

    failures = 0
    for index, row in enumerate(manifest, start=1):
        scene = row["scene"]
        pair_id = row["pair_id"]
        pair_name = row["pair_name"]
        out_dir = result_root / scene / f"pair_{pair_id}" / "traditional"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "raw.png"
        status_path = out_dir / "method_status.json"
        log_path = out_dir / "run.log"
        error_path = out_dir / "error.log"

        if not args.force and already_successful(status_path, raw_path):
            print(f"[{index}/{len(manifest)}] {pair_name}: cached")
            continue

        started = time.perf_counter()
        status = {
            "method": "traditional",
            "pair_name": pair_name,
            "success": False,
            "runtime_seconds": None,
            "failure_reason": "",
        }
        try:
            raw, info = stitch_pair(Path(row["pair_dir"]) / "0.jpg", Path(row["pair_dir"]) / "1.jpg")
            cv2.imwrite(str(raw_path), raw)
            runtime = time.perf_counter() - started
            status.update(info)
            status.update({"success": True, "runtime_seconds": runtime, "raw_path": str(raw_path)})
            log_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            error_path.write_text("", encoding="utf-8")
            print(f"[{index}/{len(manifest)}] {pair_name}: ok {runtime:.3f}s")
        except Exception as exc:
            failures += 1
            runtime = time.perf_counter() - started
            status.update({"runtime_seconds": runtime, "failure_reason": str(exc)})
            log_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[{index}/{len(manifest)}] {pair_name}: failed {exc}")
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    if failures:
        print(f"Traditional baseline completed with {failures} failed pair(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

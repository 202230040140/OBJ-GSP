import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif"}


def safe_child(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def collect_pair_images(work_root: Path) -> list[Path]:
    pairs_root = work_root / "pairs"
    if not pairs_root.exists():
        return []
    return [path for path in pairs_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]


def final_raw_exists(result_root: Path, pair_name: str, method: str) -> bool:
    if "_p" not in pair_name:
        return False
    scene, pair_id = pair_name.rsplit("_p", 1)
    return (result_root / scene / f"pair_{pair_id}" / method / "raw.png").exists()


def collect_cpp_result_images(result_root: Path, work_root: Path, method: str) -> list[Path]:
    method_root = work_root / method / "0_results"
    if not method_root.exists():
        return []
    paths = []
    for path in method_root.rglob("*.png"):
        pair_name = path.parent.name.removesuffix("-result")
        if final_raw_exists(result_root, pair_name, method):
            paths.append(path)
    return paths


def collect_debug_images(result_root: Path, work_root: Path, method: str) -> list[Path]:
    debug_root = work_root / method / "1_debugs"
    if not debug_root.exists():
        return []
    paths = []
    for path in debug_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        pair_name = path.parent.name.removesuffix("-result")
        if final_raw_exists(result_root, pair_name, method):
            paths.append(path)
    return paths


def collect_sam_debug_images(work_root: Path) -> list[Path]:
    sam_root = work_root / "obj_gsp" / "sam"
    if not sam_root.exists():
        return []
    paths = []
    for path in sam_root.rglob("*.png"):
        if (path.name in {"0-original.png", "sam.png"} or path.name.endswith("-original-runtime.png")) and (path.parent / "contour_coords.txt").exists():
            paths.append(path)
    return paths


def collect_depth_visualizations(work_root: Path) -> list[Path]:
    depth_root = work_root / "depth_assets"
    if not depth_root.exists():
        return []
    paths = []
    for path in depth_root.rglob("*-depth_vis.png"):
        depth_path = path.with_name(path.name.replace("-depth_vis.png", "-depth.png"))
        if depth_path.exists():
            paths.append(path)
    return paths


def remove_empty_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda item: len(item.parts), reverse=True):
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass
    try:
        root.rmdir()
        removed += 1
    except OSError:
        pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove redundant HD3D _work images after final results have been saved.")
    parser.add_argument("--result-root", default=r"D:\HD3D_Result")
    parser.add_argument("--work-root", default=r"D:\HD3D_Result\_work")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    work_root = Path(args.work_root)
    if not work_root.exists():
        print(f"Work root does not exist: {work_root}")
        return 0

    groups = {
        "pair_input_images": collect_pair_images(work_root),
        "obj_gsp_work_results": collect_cpp_result_images(result_root, work_root, "obj_gsp"),
        "depth_gsp_work_results": collect_cpp_result_images(result_root, work_root, "depth_gsp"),
        "obj_gsp_debug_images": collect_debug_images(result_root, work_root, "obj_gsp"),
        "depth_gsp_debug_images": collect_debug_images(result_root, work_root, "depth_gsp"),
        "sam_debug_images": collect_sam_debug_images(work_root),
        "depth_visualizations": collect_depth_visualizations(work_root),
    }

    removed_count = 0
    removed_bytes = 0
    for label, paths in groups.items():
        group_bytes = sum(path.stat().st_size for path in paths if path.exists())
        print(f"{label}: {len(paths)} file(s), {group_bytes / (1024 * 1024):.2f} MB")
        if args.dry_run:
            continue
        for path in paths:
            if not safe_child(path, work_root):
                raise RuntimeError(f"Refusing to delete outside work root: {path}")
            try:
                size = path.stat().st_size
                path.unlink()
                removed_count += 1
                removed_bytes += size
            except FileNotFoundError:
                pass

    removed_dirs = 0
    if not args.dry_run:
        removed_dirs += remove_empty_dirs(work_root / "pairs")
        removed_dirs += remove_empty_dirs(work_root / "obj_gsp" / "0_results")
        removed_dirs += remove_empty_dirs(work_root / "depth_gsp" / "0_results")

    action = "Would remove" if args.dry_run else "Removed"
    print(f"{action} {removed_count} file(s), {removed_bytes / (1024 * 1024):.2f} MB, {removed_dirs} empty dir(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


SCENE_RE = re.compile(r"^(Indoor|Outdoor)_(\d{3})_(1|2|3|4|gt)\.jpg$", re.IGNORECASE)
PAIR_ROLES = (("1", "2"), ("1", "3"), ("1", "4"), ("2", "3"), ("2", "4"), ("3", "4"))


def parse_scene_files(data_root: Path) -> dict[str, dict[str, Path]]:
    scenes: dict[str, dict[str, Path]] = {}
    for path in sorted(data_root.glob("*.jpg"), key=lambda item: item.name.lower()):
        match = SCENE_RE.match(path.name)
        if not match:
            continue
        category, scene_id, role = match.groups()
        scene = f"{category.capitalize()}_{scene_id}"
        scenes.setdefault(scene, {})[role.lower()] = path
    return scenes


def write_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "{center_image_index | 0 | center image index}",
                "{center_image_rotation_angle | 0 | center image rotation angle}",
                "{images_count | 2 | images count}",
                "{matching_graph_image_edges-1 | 0 | matching graph image edge 1}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_input(src: Path, dst: Path, force: bool, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if force:
            dst.unlink()
        else:
            try:
                if dst.samefile(src):
                    return "existing-link"
            except OSError:
                pass
            if dst.stat().st_size == src.stat().st_size:
                return "existing"

    modes = ("symlink", "hardlink", "copy") if link_mode == "auto" else (link_mode,)
    last_error = None
    for mode in modes:
        try:
            if mode == "symlink":
                os.symlink(src, dst)
                return "symlink"
            if mode == "hardlink":
                os.link(src, dst)
                return "hardlink"
            if mode == "copy":
                shutil.copy2(src, dst)
                return "copy"
        except OSError as exc:
            last_error = exc
            if dst.exists() or dst.is_symlink():
                dst.unlink()
    raise RuntimeError(f"Failed to materialize {src} -> {dst}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare HD3D two-view pair data for stitching experiments.")
    parser.add_argument("--data-root", default=r"D:\HD3D_Dataset")
    parser.add_argument("--result-root", default=r"D:\HD3D_Result")
    parser.add_argument("--scene", action="append", help="Limit to a scene such as Indoor_001. Can be repeated.")
    parser.add_argument("--pair", action="append", help="Limit to a pair id such as 12. Can be repeated.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--link-mode", choices=("auto", "symlink", "hardlink", "copy"), default="auto")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    result_root = Path(args.result_root)
    work_root = result_root / "_work"
    pairs_root = work_root / "pairs"
    graphs_root = work_root / "graphs"
    selected_scenes = set(args.scene or [])
    selected_pairs = set(args.pair or [])

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    scenes = parse_scene_files(data_root)
    complete_scenes = {
        scene: files
        for scene, files in scenes.items()
        if all(role in files for role in ("1", "2", "3", "4", "gt"))
    }
    if selected_scenes:
        complete_scenes = {scene: files for scene, files in complete_scenes.items() if scene in selected_scenes}

    expected_scenes = len(selected_scenes) if selected_scenes else 13
    if len(complete_scenes) != expected_scenes and not args.allow_count_mismatch:
        raise RuntimeError(f"Expected {expected_scenes} complete scenes, found {len(complete_scenes)}.")

    rows = []
    for scene in sorted(complete_scenes):
        files = complete_scenes[scene]
        for left_role, right_role in PAIR_ROLES:
            pair_id = f"{left_role}{right_role}"
            if selected_pairs and pair_id not in selected_pairs:
                continue
            pair_name = f"{scene}_p{pair_id}"
            pair_dir = pairs_root / pair_name
            graph_file = graphs_root / pair_name / f"{pair_name}-STITCH-GRAPH.txt"

            left_mode = materialize_input(files[left_role], pair_dir / "0.jpg", args.force, args.link_mode)
            right_mode = materialize_input(files[right_role], pair_dir / "1.jpg", args.force, args.link_mode)
            write_graph(graph_file)

            final_pair_dir = result_root / scene / f"pair_{pair_id}"
            final_pair_dir.mkdir(parents=True, exist_ok=True)
            rows.append(
                {
                    "pair_name": pair_name,
                    "scene": scene,
                    "pair_id": pair_id,
                    "left_role": left_role,
                    "right_role": right_role,
                    "left_source": str(files[left_role]),
                    "right_source": str(files[right_role]),
                    "gt_path": str(files["gt"]),
                    "pair_dir": str(pair_dir),
                    "graph_file": str(graph_file),
                    "final_pair_dir": str(final_pair_dir),
                    "left_materialized_as": left_mode,
                    "right_materialized_as": right_mode,
                }
            )

    expected_pairs = expected_scenes * (len(selected_pairs) if selected_pairs else len(PAIR_ROLES))
    if len(rows) != expected_pairs and not args.allow_count_mismatch:
        raise RuntimeError(f"Expected {expected_pairs} pairs, prepared {len(rows)}.")

    work_root.mkdir(parents=True, exist_ok=True)
    datasets_file = work_root / "datasets.txt"
    datasets_file.write_text("\n".join(row["pair_name"] for row in rows) + "\n", encoding="utf-8")

    manifest_csv = work_root / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    manifest_json = work_root / "manifest.json"
    manifest_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Prepared scenes={len(complete_scenes)} pairs={len(rows)}")
    print(f"pairs_root={pairs_root}")
    print(f"graphs_root={graphs_root}")
    print(f"datasets_file={datasets_file}")
    print(f"manifest_csv={manifest_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

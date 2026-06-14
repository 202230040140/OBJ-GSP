import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean
from typing import Optional


CATEGORIES = ("OBJ-GSP", "AANAP", "APAP", "CAVE", "DFW", "DHW", "GES", "LPC", "REW", "SEAGULL", "SVA", "SPHP")

PAPER_TARGETS = {
    "OBJ-GSP": {"mdr": 1.12229, "niqe": 2.54906},
    "AANAP": {"mdr": 1.05930, "niqe": 2.74965},
    "APAP": {"mdr": 1.20123, "niqe": 3.39280},
    "CAVE": {"mdr": 0.89731, "niqe": 4.01565},
    "DFW": {"mdr": 0.97259, "niqe": 5.69104},
    "DHW": {"mdr": 1.00496, "niqe": 2.60825},
    "GES": {"mdr": 0.98288, "niqe": 3.70041},
    "LPC": {"mdr": 1.10622, "niqe": 3.23057},
    "REW": {"mdr": 1.08635, "niqe": 2.81480},
    "SEAGULL": {"mdr": 1.08296, "niqe": 4.08903},
    "SVA": {"mdr": 1.47813, "niqe": 6.96149},
    "SPHP": {"mdr": 1.07699, "niqe": 2.49712},
}


def category_for(dataset: str) -> Optional[str]:
    for category in CATEGORIES:
        if dataset.startswith(category):
            return category
    return None


def read_datasets(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_rmse(path: Path) -> float:
    if not path.exists():
        return math.nan
    match = re.search(r"RMSE:\s*([-+0-9.eE]+)", path.read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1)) if match else math.nan


def parse_warping(path: Path) -> tuple[float, float]:
    if not path.exists():
        return math.nan, math.nan
    for line in reversed(path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        parts = line.split()
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
    return math.nan, math.nan


def load_niqe_metric(device: str):
    import pyiqa

    return pyiqa.create_metric("niqe", device=device)


def compute_niqe(metric, image_path: Path) -> float:
    if not image_path.exists():
        return math.nan
    try:
        score = metric(str(image_path))
    except Exception:
        import cv2
        import numpy as np
        import torch

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return math.nan
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        score = metric(tensor)
    return float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def pass_status(value: float, target: float, *, allow_abs: Optional[float] = None) -> str:
    if not math.isfinite(value):
        return "Missing"
    relative_ok = abs(value - target) / target <= 0.15
    absolute_ok = allow_abs is not None and abs(value - target) <= allow_abs
    return "Pass" if relative_ok or absolute_ok else "Needs Review"


def format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.5f}"


def suffix_for_method(method: str) -> str:
    method = method.lower()
    if method in {"depth", "depth-only", "depth-gsp"}:
        return "Depth-GSP_"
    if method == "gsp":
        return "GSP_"
    if method == "ges-gsp":
        return "GES-GSP_"
    return "Ours-SAM_"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OBJ-GSP StitchBench General results.")
    parser.add_argument("--experiment-root", default="experiments/phase1_depth_loss/baselines/obj_gsp_sam_general")
    parser.add_argument("--datasets-file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-niqe", action="store_true")
    parser.add_argument("--method", default="obj-gsp")
    parser.add_argument("--result-suffix")
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root)
    datasets_file = Path(args.datasets_file) if args.datasets_file else experiment_root / "datasets.txt"
    datasets = read_datasets(datasets_file)
    result_suffix = args.result_suffix or suffix_for_method(args.method)

    metric = None
    if not args.skip_niqe:
        metric = load_niqe_metric(args.device)

    per_pair_rows = []
    for dataset in datasets:
        category = category_for(dataset) or ""
        result_path = experiment_root / "0_results" / f"{dataset}-result" / f"{dataset}-{result_suffix}.png"
        debug_dir = experiment_root / "1_debugs" / f"{dataset}-result"
        rmse_path = debug_dir / f"{dataset}-RMSE-[DPS].txt"
        residual_path = debug_dir / f"{dataset}-W_Residual-[DPS].txt"

        residual_avg, residual_sd = parse_warping(residual_path)
        niqe = compute_niqe(metric, result_path) if metric is not None else math.nan
        per_pair_rows.append(
            {
                "dataset": dataset,
                "category": category,
                "result_image": str(result_path),
                "mdr_rmse": parse_rmse(rmse_path),
                "warping_residual_avg": residual_avg,
                "warping_residual_sd": residual_sd,
                "niqe": niqe,
                "status": "ok" if result_path.exists() else "missing_result",
            }
        )

    by_category_rows = []
    comparison_rows = []
    for category in CATEGORIES:
        rows = [row for row in per_pair_rows if row["category"] == category]
        mdr_values = [row["mdr_rmse"] for row in rows]
        niqe_values = [row["niqe"] for row in rows]
        valid_mdr_count = len([value for value in mdr_values if math.isfinite(value)])
        valid_niqe_count = len([value for value in niqe_values if math.isfinite(value)])
        mdr_mean = finite_mean(mdr_values)
        niqe_mean = finite_mean(niqe_values)
        residual_avg_mean = finite_mean([row["warping_residual_avg"] for row in rows])
        residual_sd_mean = finite_mean([row["warping_residual_sd"] for row in rows])
        target = PAPER_TARGETS[category]
        mdr_status = pass_status(mdr_mean, target["mdr"])
        niqe_status = pass_status(niqe_mean, target["niqe"], allow_abs=0.5)
        overall = "Pass" if mdr_status == "Pass" and niqe_status == "Pass" else "Needs Review"
        if mdr_status == "Missing" or niqe_status == "Missing":
            overall = "Missing"

        by_category_rows.append(
            {
                "category": category,
                "total_count": len(rows),
                "valid_mdr_count": valid_mdr_count,
                "valid_niqe_count": valid_niqe_count,
                "mdr_rmse_mean": mdr_mean,
                "warping_residual_avg_mean": residual_avg_mean,
                "warping_residual_sd_mean": residual_sd_mean,
                "niqe_mean": niqe_mean,
            }
        )
        comparison_rows.append(
            {
                "category": category,
                "total_count": len(rows),
                "valid_mdr_count": valid_mdr_count,
                "valid_niqe_count": valid_niqe_count,
                "paper_mdr": target["mdr"],
                "ours_mdr": mdr_mean,
                "mdr_relative_error": abs(mdr_mean - target["mdr"]) / target["mdr"] if math.isfinite(mdr_mean) else math.nan,
                "mdr_status": mdr_status,
                "paper_niqe": target["niqe"],
                "ours_niqe": niqe_mean,
                "niqe_relative_error": abs(niqe_mean - target["niqe"]) / target["niqe"] if math.isfinite(niqe_mean) else math.nan,
                "niqe_abs_error": abs(niqe_mean - target["niqe"]) if math.isfinite(niqe_mean) else math.nan,
                "niqe_status": niqe_status,
                "overall_status": overall,
            }
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: format_float(value) if isinstance(value, float) else value for key, value in row.items()})

    write_csv(experiment_root / "per_pair.csv", per_pair_rows)
    write_csv(experiment_root / "by_category.csv", by_category_rows)
    write_csv(experiment_root / "paper_comparison.csv", comparison_rows)

    report_lines = [
        f"# {args.method} StitchBench General Report",
        "",
        f"Result suffix: `{result_suffix}`",
        "",
        "| Category | Valid/Total | Paper MDR | Ours MDR | MDR Status | Paper NIQE | Ours NIQE | NIQE Status | Overall |",
        "|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in comparison_rows:
        report_lines.append(
            "| {category} | {valid_count}/{total_count} | {paper_mdr:.5f} | {ours_mdr} | {mdr_status} | {paper_niqe:.5f} | {ours_niqe} | {niqe_status} | {overall_status} |".format(
                category=row["category"],
                valid_count=row["valid_mdr_count"],
                total_count=row["total_count"],
                paper_mdr=row["paper_mdr"],
                ours_mdr=format_float(row["ours_mdr"]),
                mdr_status=row["mdr_status"],
                paper_niqe=row["paper_niqe"],
                ours_niqe=format_float(row["ours_niqe"]),
                niqe_status=row["niqe_status"],
                overall_status=row["overall_status"],
            )
        )
    report_lines.extend(
        [
            "",
            f"MDR is read from the C++ RMSE output. NIQE is computed on `*-{result_suffix}.png` with pyiqa.",
            "Automatic graph files are generated from sorted image order, so this is a loose reproduction check rather than an official-number reproduction.",
        ]
    )
    (experiment_root / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote evaluation files to {experiment_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Optional


def parse_float(value: str) -> float:
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def load_per_pair(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["mdr_rmse"] = parse_float(row.get("mdr_rmse"))
            row["warping_residual_avg"] = parse_float(row.get("warping_residual_avg"))
            row["warping_residual_sd"] = parse_float(row.get("warping_residual_sd"))
            row["niqe"] = parse_float(row.get("niqe"))
            rows[row["dataset"]] = row
    return rows


def is_ok(row: Optional[dict]) -> bool:
    return bool(
        row
        and row.get("status") == "ok"
        and math.isfinite(row.get("mdr_rmse", math.nan))
        and math.isfinite(row.get("niqe", math.nan))
    )


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def fmt(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.5f}"


def pct(value: float) -> str:
    return "" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            out = {}
            for key, value in row.items():
                out[key] = fmt(value) if isinstance(value, float) else value
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two StitchBench per_pair.csv files.")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--baseline-name", default="OBJ-GSP")
    parser.add_argument("--candidate-name", default="DepthPro-DSP")
    parser.add_argument("--output-root")
    parser.add_argument("--target-both-rate", type=float, default=0.90)
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    candidate_root = Path(args.candidate_root)
    output_root = Path(args.output_root) if args.output_root else candidate_root

    baseline = load_per_pair(baseline_root / "per_pair.csv")
    candidate = load_per_pair(candidate_root / "per_pair.csv")
    names = sorted(set(baseline) | set(candidate))

    rows = []
    common_ok = []
    baseline_failed_candidate_ok = []
    candidate_failed = []
    for name in names:
        base = baseline.get(name)
        cand = candidate.get(name)
        base_ok = is_ok(base)
        cand_ok = is_ok(cand)
        if base_ok and cand_ok:
            common_ok.append(name)
        elif not base_ok and cand_ok:
            baseline_failed_candidate_ok.append(name)
        elif not cand_ok:
            candidate_failed.append(name)

        category = (cand or base or {}).get("category", "")
        base_mdr = base["mdr_rmse"] if base else math.nan
        cand_mdr = cand["mdr_rmse"] if cand else math.nan
        base_niqe = base["niqe"] if base else math.nan
        cand_niqe = cand["niqe"] if cand else math.nan
        mdr_delta = cand_mdr - base_mdr if base_ok and cand_ok else math.nan
        niqe_delta = cand_niqe - base_niqe if base_ok and cand_ok else math.nan
        rows.append(
            {
                "dataset": name,
                "category": category,
                "baseline_status": base.get("status", "missing") if base else "missing",
                "candidate_status": cand.get("status", "missing") if cand else "missing",
                "baseline_mdr": base_mdr,
                "candidate_mdr": cand_mdr,
                "mdr_delta": mdr_delta,
                "mdr_better": int(mdr_delta < 0) if math.isfinite(mdr_delta) else "",
                "baseline_niqe": base_niqe,
                "candidate_niqe": cand_niqe,
                "niqe_delta": niqe_delta,
                "niqe_better": int(niqe_delta < 0) if math.isfinite(niqe_delta) else "",
                "both_better": int(mdr_delta < 0 and niqe_delta < 0) if math.isfinite(mdr_delta) and math.isfinite(niqe_delta) else "",
                "candidate_warping_residual_avg": cand["warping_residual_avg"] if cand else math.nan,
                "candidate_result_image": cand.get("result_image", "") if cand else "",
                "baseline_result_image": base.get("result_image", "") if base else "",
            }
        )

    category_rows = []
    by_category: dict[str, list[str]] = defaultdict(list)
    for name in common_ok:
        category = (candidate.get(name) or baseline.get(name)).get("category", "")
        by_category[category].append(name)
    for category in sorted(by_category):
        category_names = by_category[category]
        mdr_better = [name for name in category_names if candidate[name]["mdr_rmse"] < baseline[name]["mdr_rmse"]]
        niqe_better = [name for name in category_names if candidate[name]["niqe"] < baseline[name]["niqe"]]
        both_better = [
            name
            for name in category_names
            if candidate[name]["mdr_rmse"] < baseline[name]["mdr_rmse"] and candidate[name]["niqe"] < baseline[name]["niqe"]
        ]
        category_rows.append(
            {
                "category": category,
                "common_ok_count": len(category_names),
                "candidate_mdr_mean": finite_mean([candidate[name]["mdr_rmse"] for name in category_names]),
                "baseline_mdr_mean": finite_mean([baseline[name]["mdr_rmse"] for name in category_names]),
                "candidate_niqe_mean": finite_mean([candidate[name]["niqe"] for name in category_names]),
                "baseline_niqe_mean": finite_mean([baseline[name]["niqe"] for name in category_names]),
                "mdr_better_count": len(mdr_better),
                "mdr_better_rate": len(mdr_better) / len(category_names),
                "niqe_better_count": len(niqe_better),
                "niqe_better_rate": len(niqe_better) / len(category_names),
                "both_better_count": len(both_better),
                "both_better_rate": len(both_better) / len(category_names),
            }
        )

    common_candidate_mdr = finite_mean([candidate[name]["mdr_rmse"] for name in common_ok])
    common_baseline_mdr = finite_mean([baseline[name]["mdr_rmse"] for name in common_ok])
    common_candidate_niqe = finite_mean([candidate[name]["niqe"] for name in common_ok])
    common_baseline_niqe = finite_mean([baseline[name]["niqe"] for name in common_ok])
    both_better = [
        name
        for name in common_ok
        if candidate[name]["mdr_rmse"] < baseline[name]["mdr_rmse"] and candidate[name]["niqe"] < baseline[name]["niqe"]
    ]
    mdr_better = [name for name in common_ok if candidate[name]["mdr_rmse"] < baseline[name]["mdr_rmse"]]
    niqe_better = [name for name in common_ok if candidate[name]["niqe"] < baseline[name]["niqe"]]
    both_rate = len(both_better) / len(common_ok) if common_ok else math.nan

    write_csv(output_root / "method_pair_comparison.csv", rows)
    write_csv(output_root / "method_category_comparison.csv", category_rows)

    regressions = []
    for row in rows:
        if row["both_better"] == 1:
            continue
        if not math.isfinite(row["mdr_delta"]) or not math.isfinite(row["niqe_delta"]):
            continue
        severity = max(row["mdr_delta"], 0.0) + max(row["niqe_delta"], 0.0)
        regressions.append((severity, row))
    regressions.sort(key=lambda item: item[0], reverse=True)

    report = [
        f"# {args.candidate_name} vs {args.baseline_name}",
        "",
        "## Summary",
        "",
        f"- Total datasets: {len(names)}",
        f"- Common successful datasets: {len(common_ok)}",
        f"- {args.baseline_name} failed while {args.candidate_name} succeeded: {len(baseline_failed_candidate_ok)}",
        f"- {args.candidate_name} failed: {len(candidate_failed)}",
        f"- Common mean MDR: {args.candidate_name} {fmt(common_candidate_mdr)} vs {args.baseline_name} {fmt(common_baseline_mdr)}",
        f"- Common mean NIQE: {args.candidate_name} {fmt(common_candidate_niqe)} vs {args.baseline_name} {fmt(common_baseline_niqe)}",
        f"- MDR better on common set: {len(mdr_better)}/{len(common_ok)} ({pct(len(mdr_better) / len(common_ok) if common_ok else math.nan)})",
        f"- NIQE better on common set: {len(niqe_better)}/{len(common_ok)} ({pct(len(niqe_better) / len(common_ok) if common_ok else math.nan)})",
        f"- Both MDR and NIQE better on common set: {len(both_better)}/{len(common_ok)} ({pct(both_rate)})",
        f"- Target both-better rate: {pct(args.target_both_rate)}",
        f"- Target status: {'Pass' if both_rate >= args.target_both_rate and common_candidate_mdr < common_baseline_mdr and common_candidate_niqe < common_baseline_niqe else 'Needs Optimization'}",
        "",
        "## Category Breakdown",
        "",
        "| Category | N | Candidate MDR | Baseline MDR | Candidate NIQE | Baseline NIQE | Both Better |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in category_rows:
        report.append(
            f"| {row['category']} | {row['common_ok_count']} | {fmt(row['candidate_mdr_mean'])} | {fmt(row['baseline_mdr_mean'])} | "
            f"{fmt(row['candidate_niqe_mean'])} | {fmt(row['baseline_niqe_mean'])} | "
            f"{row['both_better_count']}/{row['common_ok_count']} ({pct(row['both_better_rate'])}) |"
        )

    report.extend(["", "## Baseline Failed, Candidate Succeeded", ""])
    if baseline_failed_candidate_ok:
        for name in baseline_failed_candidate_ok:
            report.append(f"- {name}")
    else:
        report.append("- None")

    report.extend(["", "## Largest Common-Set Regressions", ""])
    if regressions:
        report.append("| Dataset | MDR Delta | NIQE Delta | Candidate Warp Avg | Candidate Result |")
        report.append("|---|---:|---:|---:|---|")
        for _, row in regressions[:30]:
            report.append(
                f"| {row['dataset']} | {fmt(row['mdr_delta'])} | {fmt(row['niqe_delta'])} | "
                f"{fmt(row['candidate_warping_residual_avg'])} | `{row['candidate_result_image']}` |"
            )
    else:
        report.append("- None")

    (output_root / "method_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote comparison files to {output_root}")
    print(f"common_ok={len(common_ok)} both_better={len(both_better)} both_rate={pct(both_rate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# StitchBench General 上复现 OBJ-GSP Ours 实验计划

## Summary
- 目标：只复现论文表格中的 **OBJ-GSP (ours)**，在 `D:\StitchBench\General` 的 12 类数据集上批量运行，并汇总 MDR/RMSE 与 NIQE，对照论文中 OBJ-GSP 行判断是否达到宽松复现效果。
- 数据范围：包含表格里的 `OBJ-GSP/AANAP/APAP/CAVE/DFW/DHW/GES/LPC/REW/SEAGULL/SVA/SPHP` 共 100 组；排除额外的 `NISwGSP-*` 目录。
- 当前数据集没有官方 `*-STITCH-GRAPH.txt`，采用自动 graph：图像按文件名排序，两图互连；多图按顺序链路连接，中心图为 0。
- `D:\StitchBench\General` 只读使用；所有 graph、SAM 轮廓、结果图、指标表写入 `experiments/stitchbench_general_ours/`。

## Key Changes
- 批量化 C++ 入口：
  - 给 `obj_gsp.exe` 增加参数：`--data-root`、`--graph-root`、`--sam-root`、`--output-root`、`--datasets-file`。
  - `Parameter` 从 `data-root/<dataset>` 读图片，从 `graph-root/<dataset>/<dataset>-STITCH-GRAPH.txt` 读自动 graph，从 `output-root` 写结果和 debug。
  - `ImageData` 从 `sam-root/<dataset>/contour_coords.txt` 读 SAM 轮廓，不再使用仓库根目录的固定 `contour_coords.txt`。
  - 图片文件名排序固定化，并修正图片扩展名列表里的 `.PNG` 拼接问题，保证批量结果可复现。
- 新增批量脚本：
  - `tools/prepare_stitchbench_general.py`：发现 12 类数据集，生成 `datasets.txt`、manifest、自动 graph。
  - `tools/generate_sam_assets.py`：使用现有 `obj-gsp-sam` 环境和 `sam_vit_h_4b8939.pth`，按原 notebook 逻辑为每组生成 `0-original.png`、`sam.png`、`contour_coords.txt`；默认缓存已有结果。
  - `tools/evaluate_stitchbench_ours.py`：解析 C++ 输出的 RMSE 作为 MDR，保留 warping residual avg/sd，使用 `pyiqa` 计算 NIQE，生成 `per_pair.csv`、`by_category.csv`、`paper_comparison.csv` 和 `report.md`。
  - `tools/run_stitchbench_ours.ps1`：统一执行准备、SAM、CMake Release 构建、批量运行、评估汇总。
- NIQE 依赖：
  - 在 `C:\Users\22499\.venvs\obj-gsp-sam` 中安装 `pyiqa`，记录版本；本机未检测到 MATLAB/Octave，因此不采用 MATLAB NIQE。

## Test Plan
- Smoke test：先跑 `AANAP-01_skyline`，确认仍得到结果图，且 RMSE 约接近已验证的 `1.06956`。
- 数据准备验收：manifest 中必须有 100 组；每组 graph 文件存在；每组图片数与数据目录一致。
- SAM 验收：100 组均生成 `contour_coords.txt`；为空或轮廓数过少的组标记为 `sam_warning`，但不中断整体实验。
- C++ 验收：100 组均生成 `*-Ours-SAM_.png`；失败组写入 `failed_runs.csv` 并保留 stdout/stderr。
- 指标验收：每类输出 MDR/RMSE 均值和 NIQE 均值；与论文 OBJ-GSP 行对比，MDR/NIQE 误差在 `15%` 内或 NIQE 绝对差 `<=0.5` 记为宽松达标，超出则在报告中标为 `Needs Review` 并附对应结果图路径。

## Assumptions
- 只复现 OBJ-GSP ours，不复现 GSP、GES-GSP、APAP、UDIS 等 baseline。
- 自动 graph 不是官方 graph，因此结论是“宽松复现/效果核查”，不声明完全复刻论文数值。
- SAM 使用 ViT-H、CUDA、本地 checkpoint；沿用 notebook 的 `SamAutomaticMaskGenerator(min_mask_region_area=10000)`。
- 论文表中的 MDR 用当前代码输出的 `RMSE` 对照；warping residual avg/sd 作为辅助诊断，不作为表格 MDR 主指标。

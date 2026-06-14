# Phase 1 Depth-GSP Current Results

This report summarizes the current fair first-stage geometry replacement run. It does not use seam repair, cropping, smoothing, denoising, inpainting, or per-sample output selection.

## Current Fixed Setting

Command entry:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\run_phase1_depth_loss.ps1
```

Effective fixed parameters:

- `ContentWeight = 0.75`
- `DepthTau = 0.25`
- `DepthCrossLayerWeight = 0.05`
- `DepthMinWeight = 0.02`
- `DepthConfidenceFloor = 0.10`
- `DepthStructureWeight = 0.25`
- `DepthTextureWeight = 0.10`
- `DepthEdgeWeight = 0.10`
- `DepthTextureNoiseWeight = 0.75`
- `DepthPlanarityWeight = 0.35`

DepthPro assets are read from:

```text
assets/depthpro/stitchbench_general/
```

## Verification Against OBJ-GSP

Candidate run:

```text
experiments/phase1_depth_loss/runs/depth_gsp_v5_planarity035/
```

Baseline:

```text
experiments/phase1_depth_loss/baselines/obj_gsp_sam_general/
```

Common valid samples: 85.

| Metric | OBJ-GSP | Depth-GSP v5 | Relative Gap |
|---|---:|---:|---:|
| Mean MDR | 1.06828 | 1.06318 | -0.48% |
| Mean NIQE | 3.49224 | 3.23966 | -7.23% |

Safety checks:

- Depth-GSP v5 failed samples: 0.
- Max MDR ratio: 1.32961 on `OBJ-GSP-tree2`.
- Max NIQE ratio: 1.19083 on `GES-13-Farmland-1`.
- Samples with metric ratio >= 2.0: 0.

Interpretation:

- The strict MDR parity target is met.
- NIQE is not within a strict absolute 1% parity band; it is substantially better, not worse.
- If the acceptance rule means "not worse than OBJ-GSP by more than 1%", the current fixed first-stage `psi_depth` passes.
- If the rule means exact absolute parity within 1% even when better, NIQE remains outside that definition.
- A lower `ContentWeight=0.10` variant moved NIQE closer (`-3.4%`) and improved MDR (`-5.1%`), but failed on `OBJ-GSP-farm`; it is rejected because the no-crash requirement is stricter than matching NIQE by intentionally weakening the geometry term.

## Loss Revision Notes

The current `psi_depth` keeps the original OBJ-GSP role of protecting meaningful geometry, but replaces SAM object contours with DepthPro-derived structure:

- Same-layer and depth-continuous mesh edges are preserved more strongly.
- Cross-layer edges are weakly coupled to avoid binding foreground and background.
- Fused depth/texture structure edges provide a small boost.
- Texture-only edge noise is penalized, which reduces failures on foliage and repetitive texture.

Key iteration findings:

- Weak depth constraints (`ContentWeight=0.10`) gave good mean metrics but failed on `OBJ-GSP-farm`.
- Strong constraints (`ContentWeight=1.00`) removed crashes but caused a 2x MDR regression on `OBJ-GSP-tree2`.
- Adding texture-only noise penalty reduced `OBJ-GSP-tree2` MDR ratio from 1.49 to 1.35 while preserving 100/100 success.
- Adding local depth-planarity gating further reduced `OBJ-GSP-tree2` MDR ratio from 1.35 to 1.33 and improved MDR on 75/85 common valid samples.
- Low-content strict-parity probes showed that trying to force NIQE closer to OBJ-GSP weakens the geometry term enough to revive the `OBJ-GSP-farm` failure. This supports keeping the stable non-worse setting as the fair first-stage replacement result.

## Remaining Weak Cases

- `OBJ-GSP-tree2`: dense foliage; depth and texture are noisy, so even downweighted constraints still over-regularize.
- `LPC-06`: strong parallax and architecture lines; depth-guided local structure still lags SAM object constraints in MDR.
- `OBJ-GSP-02_food`: smooth tabletop and curved object boundaries; NIQE is the largest remaining ratio, though still far below 2x.

Next useful change:

- Diagnose the remaining `LPC-06` and `OBJ-GSP-02_food` cases with layer-wise homography/planarity statistics before changing the loss again.

# Phase 1 Depth Loss Research Plan

This phase tests only the geometry-stage replacement:

```text
OBJ-GSP:   alignment + similarity + psi_obj
Depth-GSP: alignment + similarity + psi_depth
```

No seam repair, image cropping, smoothing, denoising, inpainting, or per-sample output selection is allowed in the final comparison.

## Fair Comparison Rule

- Use the same input datasets, graph generation, matching, mesh, warping, and blending code.
- Only replace `psi_obj(V)` with `psi_depth(V)`.
- Use one fixed `psi_depth` definition and one fixed parameter set for the final run.
- Development sweeps may inspect MDR/NIQE, but final reported results must not select outputs per sample.

## Acceptance Criteria

Evaluate only samples where both OBJ-GSP and Depth-GSP produce valid results:

- Mean MDR non-worse relative gap <= 1%, where gap is `Depth-GSP / OBJ-GSP - 1`.
- Mean NIQE non-worse relative gap <= 1%, using the same definition.
- Depth-GSP has no crashes.
- No sample has `Depth-GSP metric >= 2 * OBJ-GSP metric` for MDR or NIQE.

The report also records a strict absolute mean-gap check. That check is diagnostic only: if Depth-GSP is substantially better on NIQE, it can fail absolute parity while still satisfying the non-worse reproduction target.

## Current psi_depth Direction

DepthPro assets are centralized under:

```text
assets/depthpro/stitchbench_general/
```

For each input image, asset generation stores:

- `*-depth.png`
- `*-depth_layers.png`
- `*-depth_conf.png`
- `*-depth_edges.png`
- `*-texture_edges.png`
- `*-structure_edges.png`
- `*-depth_vis.png`

The intended loss mirrors the role of `psi_obj`: preserve meaningful geometric structures, not arbitrary dense pixels. Depth layers and confidence decide where a structure is reliable; texture and depth edges decide where the structure matters.

Current fixed edge weight:

```text
psi_depth(e) = w_content * w_depth(e) * ||(V_j - V_i) - (P_j - P_i)||^2

w_depth(e) = clamp(
    w_layer * w_continuity * w_confidence
    * w_structure_boost
    * w_texture_noise_penalty
    * w_planarity_penalty,
    depth_min_weight,
    3.0
)
```

Where:

- `w_layer` is high inside one depth layer and low across layers.
- `w_continuity = exp(-abs(d_i - d_j) / tau)`.
- `w_structure_boost` uses fused structure, texture, and depth edge strengths.
- `w_texture_noise_penalty` suppresses texture-only edges without depth support.
- `w_planarity_penalty` suppresses locally non-planar or noisy depth patches.

## Iteration Loop

1. Run fixed-parameter Depth-GSP on all 100 StitchBench General groups.
2. Compare against the retained OBJ-GSP baseline under `experiments/phase1_depth_loss/baselines/obj_gsp_sam_general/`.
3. Sort failures by MDR/NIQE relative gap.
4. Update only `psi_depth` weights or structure sampling.
5. Re-run the same fixed setting until acceptance is met or a clear blocker appears.

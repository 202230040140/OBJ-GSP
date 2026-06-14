# Local Weights

Large model files are kept here for local reproduction and are ignored by Git.

Expected local layout:

```text
weights/sam/sam_vit_h_4b8939.pth
```

DepthPro is currently loaded from the local Hugging Face snapshot:

```text
D:\HFModels\DepthPro-hf
```

You can also place a local copy under `weights/depthpro/DepthPro-hf/` and pass that path with `--depthpro-model`.


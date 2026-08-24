# Prior-art matrix

| Work | Intended user/problem | Primary signal | Strength | Gap relevant to Continuity Lens |
|---|---|---|---|---|
| [V-JEPA 2](https://github.com/facebookresearch/vjepa2) | Learned video representations and prediction | Masked latent prediction | Encodes space-time structure without pixel reconstruction | Not trained or calibrated as an editing-boundary quality score |
| [VBench](https://github.com/Vchitect/VBench) | Multi-dimensional evaluation of generated video | Subject/background consistency, motion, quality metrics | Broad and interpretable evaluation suite | Suite-level scoring does not isolate whether learned future prediction adds boundary-level value |
| [TransNet V2](https://github.com/soCzech/TransNetV2) | Shot-boundary detection | Learned cut probability | Strong at finding edits | A valid cut detector does not determine whether a transition is physically or narratively coherent |
| SSIM and HSV histograms | Pixel/appearance similarity | Local structure and color-distribution change | Fast, cheap, understandable | Conflate intentional appearance change with continuity failure |
| Farneback optical flow | Motion between adjacent frames | Dense pixel displacement | Cheap temporal signal | Brittle under camera motion, occlusion, texture loss, and generated artifacts |
| [VideoScore](https://github.com/TIGER-AI-Lab/VideoScore) | Learned text-to-video quality assessment | Multi-aspect learned quality | Captures higher-level generation quality | Heavier, broader, and not designed to explain one edit boundary |

## Research gap

The useful question is not whether V-JEPA can produce another scalar. It is whether its masked-future prediction error adds held-out discrimination beyond signals a product team could implement cheaply—and whether any gain justifies its latency and complexity.

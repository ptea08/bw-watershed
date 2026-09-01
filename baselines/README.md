# Baselines

How the comparison rows of Table 1 were produced. The scripts themselves are not
in this repository — each baseline has **mutually conflicting dependencies**
(StarDist pulls TensorFlow, Mask2Former pulls detectron2) and was run in its own
environment, off the main pipeline. What follows is the configuration each row
used, so the numbers can be reproduced from the upstream projects directly.

| Baseline | Table 1 row | Setting |
|---|---|---|
| EcoTaxa / ZooProcess threshold | `EcoTaxa` | Global intensity threshold + connected components. Binary-mask reference with no explicit separation of touching organisms. |
| CellPose (native) | `CellPose (native)` | Trained directly on the same 3 annotated crops, 50 epochs to convergence. |
| StarDist (native) | `StarDist (native)` | Trained directly on the same 3 annotated crops, 50 epochs to convergence. |
| CellPose + RF mask | instance-separation ablation | Same RF-generated pseudo-labels as the proposed method; only the instance-recovery step differs. |
| StarDist + RF mask | instance-separation ablation | As above. |
| Cellpose-SAM | `CP-SAM (ZS)` | Zero-shot, evaluated without fine-tuning. |
| Mask2Former (Swin-T) | `M2F (Swin-T)` | COCO-pretrained, no ZooScan-specific adaptation. |

The instance-separation ablations hold the pseudo-label source and training crop
pool fixed while changing only how instances are recovered — that isolates the
contribution of boundary subtraction + watershed from the contribution of
bootstrapping.

Every row is scored the same way as the proposed method — one-to-one matching at
IoU 0.5, macro-averaged over the same 15 held-out crops — so the comparison
turns on the segmentation, not on the metric.

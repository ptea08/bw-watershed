"""Bootstrapped Watershed — few-shot instance segmentation for ZooScan images.

Three stages, mirroring Figure 1 of the paper:

1. ``stage1_bootstrap``  — frozen DINOv3 features + a shallow MLP trained on a
   handful of annotated crops, used to pseudo-label the unlabeled crops it is
   most confident about.
2. ``stage2_segmenter``  — a U-Net decoder on the frozen backbone, supervised
   by those pseudo-labels, predicting foreground and boundary maps.
3. ``stage3_instances``  — tiled full-scan inference, then boundary removal,
   pinch-point severing and distance-transform watershed to recover instances.
"""

__version__ = "0.1.0"

"""Full-scan tiled inference.

Paper, Sec. 3.2: a full scan contains approximately 15000 x 25000 pixels and
cannot be processed at once with the available GPU memory. Inference therefore
runs on 512 x 512 tiles overlapping by 128 px, with Gaussian-weighted blending
across overlaps (see ``data.tiling``).

Blending happens in logit space rather than on argmax labels. Averaging class
decisions across a seam produces a visible discontinuity wherever two tiles
disagree; averaging logits lets a confident tile outvote an uncertain one, and
the argmax is taken once at the end over the whole scan.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..data.masks import encode_class_map
from ..data.tiling import TileBlender, gaussian_weight_map, iter_tiles
from ..stage1_bootstrap.features import pad_to_multiple
from .extraction import extract_instances


def load_model(cfg, checkpoint_path: Path, device):
    """Rebuild the segmenter and load trained decoder weights.

    Checkpoints hold decoder weights only; the frozen backbone is fetched from
    the Hub by name, so a checkpoint stays small and cannot silently ship a
    modified encoder.
    """
    from ..stage2_segmenter.decoder import ConvNeXtUNet

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ConvNeXtUNet(cfg).to(device)
    model.decode_head.load_state_dict(checkpoint["decoder_state_dict"])
    model.eval()
    return model


def preprocess_tile(tile_bgr: np.ndarray, cfg, stride: int):
    """BGR tile -> normalised NCHW tensor, padded up to the encoder stride."""
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    padded, height, width = pad_to_multiple(rgb, stride)

    mean = np.asarray(cfg.data.normalization.mean, dtype=np.float32)
    std = np.asarray(cfg.data.normalization.std, dtype=np.float32)
    normalized = (padded.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1))[None].float()
    return tensor, height, width


@torch.no_grad()
def predict_scan(model, scan_bgr: np.ndarray, cfg, device) -> np.ndarray:
    """Return the blended ``(C, H, W)`` logit map for a full scan.

    Memory note: a 15000 x 25000 float32 map is ~1.5 GB per channel, so the
    accumulator lives on the host. Only one tile at a time is ever on the GPU.
    """
    tiling = cfg.stage3_instances.tiling
    stride = cfg.stage2_segmenter.decoder.stride
    height, width = scan_bgr.shape[:2]

    blender = TileBlender(
        height,
        width,
        cfg.data.num_classes,
        gaussian_weight_map(tiling.tile_size, tiling.blend_sigma),
    )

    # Materialised so the tile count is known up front — a full scan is a few
    # thousand tuples, which is nothing beside the accumulator above, and a
    # loop with no denominator is exactly what makes this stage look hung.
    tiles = list(iter_tiles(height, width, tiling.tile_size, tiling.overlap))
    report_every = max(1, len(tiles) // 20)

    for index, bounds in enumerate(tiles, 1):
        y0, y1, x0, x1 = bounds
        tensor, tile_h, tile_w = preprocess_tile(scan_bgr[y0:y1, x0:x1], cfg, stride)
        logits = model(tensor.to(device))
        blender.add(logits[0, :, :tile_h, :tile_w].float().cpu().numpy(), bounds)

        if index % report_every == 0 or index == len(tiles):
            print(
                f"      tile {index}/{len(tiles)}  ({100 * index / len(tiles):.0f}%)",
                flush=True,
            )

    return blender.result()


def logits_to_mask(logits: np.ndarray, cfg) -> np.ndarray:
    """Argmax a logit map into the RGB semantic mask stage 3 consumes.

    The median blur removes single-pixel speckle that would otherwise become
    spurious watershed markers or punch holes in the boundary wall.
    """
    class_map = np.argmax(logits, axis=0).astype(np.uint8)
    return cv2.medianBlur(
        encode_class_map(class_map), cfg.stage3_instances.extraction.median_blur
    )


def run(model, scan_path: Path, output_dir: Path, cfg, device, save_maps: bool = False):
    """Predict one scan, extract instances, and write the results.

    Writes ``<stem>_mask.png`` and ``<stem>_instances.npy``, plus the raw
    logits when ``save_maps`` is set. Returns the instance label image.
    """
    scan_path, output_dir = Path(scan_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan = cv2.imread(str(scan_path))
    if scan is None:
        raise FileNotFoundError(scan_path)

    height, width = scan.shape[:2]
    print(f"    {scan_path.name}: {width}x{height}, tiling ...", flush=True)
    logits = predict_scan(model, scan, cfg, device)

    print("    argmax + median blur ...", flush=True)
    mask = logits_to_mask(logits, cfg)

    # Whole-image morphology, distance transform and watershed on a full scan —
    # a single silent step that can outlast the tiling loop above it.
    print("    severing pinch points and running the watershed ...", flush=True)
    instances = extract_instances(mask, scan, cfg)

    print("    writing ...", flush=True)
    cv2.imwrite(str(output_dir / f"{scan_path.stem}_mask.png"), mask)
    np.save(output_dir / f"{scan_path.stem}_instances.npy", instances)
    if save_maps:
        np.save(output_dir / f"{scan_path.stem}_logits.npy", logits.astype(np.float16))

    return instances

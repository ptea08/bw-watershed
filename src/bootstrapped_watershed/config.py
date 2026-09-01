"""Configuration loading and process-wide setup.

All pipeline constants live in ``configs/default.yaml``. Load once, pass the
resulting object down; never hardcode a threshold in a module. Paths are not
part of the config — they arrive as command-line arguments.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config(dict):
    """Dict with attribute access, so ``cfg.stage1_bootstrap.classifier.type`` works."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay ``overlay`` onto ``base``, returning a new dict.

    Nested mappings merge key by key; anything else replaces wholesale. Lists
    are replaced rather than concatenated — ``boundary_dilation: [3, 5]`` is one
    setting, not two, and appending to it would be meaningless.
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load the default config, then overlay ``path`` and ``overrides`` onto it.

    ``configs/default.yaml`` is always the base. A config passed with
    ``--config`` is an *overlay*: it needs to contain only the keys it changes,
    at any depth, and everything it omits keeps its published value. That keeps
    a user's config small enough to read as a record of how their setup differs
    from the paper, and stops it going stale when the defaults change.

    See ``docs/TUNING.md`` for which settings are worth overriding.
    """
    if not DEFAULT_CONFIG_PATH.is_file():
        # `configs/` sits beside `src/`, so the default is only found from a
        # source checkout or an editable install. A non-editable `pip install .`
        # lands the package in site-packages without it.
        raise FileNotFoundError(
            f"config not found at {DEFAULT_CONFIG_PATH}. Install with "
            f"`pip install -e .` from a clone, or pass an explicit path with "
            f"--config."
        )
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config overlay not found at {path}")
        with open(path) as f:
            # An empty overlay file parses to None, which is a legitimate
            # "change nothing" rather than an error.
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    return Config(_deep_merge(cfg, overrides))


def class_weights(cfg) -> list[float]:
    """Cross-entropy weights ordered to match the integer class indices.

    The config keys these by class *name* precisely so that this function is
    the single place where names become positions. Reordering
    ``data.class_names`` reorders the tensor automatically instead of
    silently mislabelling it.
    """
    named = cfg.data.class_weights
    return [float(named[name]) for name in cfg.data.class_names]


def class_index(cfg, name: str) -> int:
    """Integer index of a named class, e.g. ``class_index(cfg, "boundary") == 2``."""
    return list(cfg.data.class_names).index(name)


def resolve_device(cfg):
    """Honour ``cfg.device`` but degrade to CPU rather than crashing."""
    import torch

    requested = str(cfg.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def require_hf_token() -> str:
    """Return a Hugging Face token, or explain how to get one.

    Checks ``HF_TOKEN`` first, then the token stored by ``hf auth login``.
    Both are supported because the official CLI writes only the file: a user
    who logged in the documented way, and is genuinely authorised, would
    otherwise be told the repository is gated.

    The DINOv3 checkpoints are gated. There is deliberately no default value:
    a hardcoded fallback token would be committed to a public repository the
    first time someone forgot to set the variable.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except ImportError:
            token = None
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. The DINOv3 checkpoints are gated on "
            "the Hugging Face Hub: request access at "
            "https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m "
            "then either run `hf auth login` or `export HF_TOKEN=hf_...` "
            "before starting the pipeline."
        )
    return token


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch. Does not enforce cuDNN determinism."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

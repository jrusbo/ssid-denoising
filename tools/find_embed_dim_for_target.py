"""Utility script to find the optimal embedding dimension to hit a specific parameter target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config  # noqa: E402
from models.hasst import HASST  # noqa: E402
from models.mamba_ir import get_mamba_class  # noqa: E402


def count_params(embed_dim: int, cfg: Config) -> int:
    """Calculates the number of parameters for a HASST model with a given embed_dim.

    Args:
        embed_dim: The embedding dimension to test.
        cfg: The configuration object containing model architecture parameters.

    Returns:
        The total number of parameters in the model.
    """
    model = HASST(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        embed_dim=embed_dim,
        num_blocks=cfg.num_blocks,
        lonpe_scale_physical=cfg.lonpe_scale_physical,
        lonpe_shot_range=tuple(cfg.lonpe_shot_range),
        lonpe_read_range=tuple(cfg.lonpe_read_range),
    )
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    """Finds the embed_dim that results in a parameter count closest to the target.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Find embed_dim that gets closest to target parameter count."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_config_example.yaml")
    )
    parser.add_argument("--target-m", type=float, default=18.7)
    parser.add_argument("--min-embed", type=int, default=16)
    parser.add_argument("--max-embed", type=int, default=48)
    args = parser.parse_args()

    cfg = Config.load_from_yaml(args.config)
    mamba_available = bool(get_mamba_class())

    best: Optional[Tuple[float, int, float]] = None
    for d in range(args.min_embed, args.max_embed + 1):
        total_m = count_params(d, cfg) / 1e6
        diff = abs(total_m - args.target_m)
        if best is None or diff < best[0]:
            best = (diff, d, total_m)

    assert best is not None
    _, best_d, best_m = best
    signed = best_m - args.target_m

    print(f"config: {args.config}")
    print(f"num_blocks: {cfg.num_blocks}")
    print(
        f"mamba_global_branch: {'enabled' if mamba_available else 'disabled (fallback)'}"
    )
    print(f"target: {args.target_m:.4f}M")
    print(f"best_embed_dim: {best_d}")
    print(f"params_at_best: {best_m:.4f}M (diff {signed:+.4f}M)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

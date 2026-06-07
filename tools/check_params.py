"""Verification tool to ensure the HASST model parameter count remains within a specified range."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config  # noqa: E402
from models.hasst import HASST  # noqa: E402
from models.mamba_ir import get_mamba_class  # noqa: E402


def count_params(cfg: Config) -> int:
    """Calculates the total number of parameters for the HASST model based on config.

    Args:
        cfg: The configuration object containing model architecture parameters.

    Returns:
        The total number of parameters in the model.
    """
    model = HASST(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        embed_dim=cfg.embed_dim,
        num_blocks=cfg.num_blocks,
        lonpe_scale_physical=cfg.lonpe_scale_physical,
        lonpe_shot_range=tuple(cfg.lonpe_shot_range),
        lonpe_read_range=tuple(cfg.lonpe_read_range),
    )
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    """Checks the HASST parameter count against a target range and tolerance.

    Returns:
        0 if within tolerance, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Check HASST parameter count against a target range."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Optional YAML config path."
    )
    parser.add_argument(
        "--target-m",
        type=float,
        default=18.7,
        help="Target parameter count in millions.",
    )
    parser.add_argument(
        "--tol-m",
        type=float,
        default=0.3,
        help="Allowed absolute tolerance in millions.",
    )
    args = parser.parse_args()

    cfg = Config.load_from_yaml(args.config) if args.config else Config()
    mamba_available = bool(get_mamba_class())
    total = count_params(cfg)
    total_m = total / 1e6
    diff_m = total_m - args.target_m

    print(f"config: {args.config if args.config else 'Config() defaults'}")
    print(f"embed_dim={cfg.embed_dim}, num_blocks={cfg.num_blocks}")
    print(
        f"mamba_global_branch={'enabled' if mamba_available else 'disabled (fallback)'}"
    )
    print(f"params={total:,} ({total_m:.4f}M)")
    print(f"target={args.target_m:.4f}M, diff={diff_m:+.4f}M")

    if abs(diff_m) <= args.tol_m:
        print("PASS: parameter count is within tolerance")
        return 0

    print("FAIL: parameter count is outside tolerance")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

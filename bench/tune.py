"""Measure launch configurations of torchrolling's Triton kernels on this GPU.

    python bench/tune.py                    # writes bench/results/tuning.json
    python bench/bench.py --tuning bench/results/tuning.json

For every kernel and window size it tries tile sizes and warp counts (and, for quantiles,
sorting every window against the select kernel), keeps the fastest, and writes them in the
shape of the tables at the top of ``torchrolling/_triton.py``. ``apply`` loads such a file
into those tables, for bench.py; the best ones end up in the source.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import torchrolling
from torchrolling import _triton as kernels

RESULTS = Path(__file__).parent / "results"
# Representative statistics per rolling group: their times are added up.
GROUPS = {"light": ["mean", "max"], "moments": ["std"], "heavy": ["corr", "kurt"]}
BLOCK_WS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TILES = [512, 1024, 2048, 4096, 8192]
WARPS = [2, 4, 8]
QUANTILE_WINDOWS = [4, 8, 16, 32, 64, 128, 256, 1024, 4096]
SELECT_BLOCKS = [32, 64, 128, 256, 512, 1024]
EWM_BLOCKS = [256, 512, 1024, 2048]


def best_time(fn: Callable[[], object], repeats: int = 3) -> float:
    fn()  # compiles
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


def attempt(fn: Callable[[], object]) -> float:
    """Time ``fn``, or infinity when the configuration does not compile or run."""
    try:
        return best_time(fn)
    except Exception as e:  # out of resources, compiler limits, ...
        print(f"    skipped: {type(e).__name__}: {str(e).splitlines()[0][:80]}", flush=True)
        return float("inf")


def window_for(block_w: int) -> int:
    """A window that needs exactly this BLOCK_W."""
    return max(1, block_w * 3 // 4)


def tune_rolling(x: torch.Tensor, y: torch.Tensor) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for group, stats in GROUPS.items():
        out[group] = {}
        for block_w in BLOCK_WS:
            window = window_for(block_w)
            best = (float("inf"), 0, 0)
            for tile in (t for t in TILES if t >= block_w):
                for warps in WARPS:
                    kernels.ROLLING_CONFIGS[group][block_w] = (tile, warps)
                    total = 0.0
                    for stat in stats:
                        r = torchrolling.rolling(x, window)
                        args = (y,) if stat == "corr" else ()
                        total += attempt(lambda r=r, stat=stat, args=args: getattr(r, stat)(*args))
                    best = min(best, (total, tile, warps))
            kernels.ROLLING_CONFIGS[group][block_w] = best[1:]
            out[group][str(block_w)] = [best[1], best[2]]
            print(
                f"rolling {group} BLOCK_W={block_w}: {best[1:]}, {best[0] * 1e3:.2f} ms", flush=True
            )
    return out


def tune_quantile(x: torch.Tensor) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for window in QUANTILE_WINDOWS:
        options: list[tuple[str, int, int]] = []
        if window <= 128:
            options += [("sort", 0, warps) for warps in WARPS]
        options += [("select", b, w) for b in SELECT_BLOCKS for w in WARPS if b + window <= 8192]
        best = (float("inf"), ("sort", 0, 4))
        for option in options:
            kernels.QUANTILE_CONFIGS[window] = option
            t = attempt(lambda window=window: torchrolling.rolling(x, window).median())
            best = min(best, (t, option))
        kernels.QUANTILE_CONFIGS[window] = best[1]
        out[str(window)] = list(best[1])
        print(f"quantile window={window}: {best[1]}, {best[0] * 1e3:.2f} ms", flush=True)
    return out


def tune_ewm(x: torch.Tensor) -> list[int]:
    best = (float("inf"), 1024, 4)
    for block in EWM_BLOCKS:
        for warps in WARPS:
            kernels.EWM_CONFIG = (block, warps)
            e = torchrolling.ewm(x, span=50)
            t = attempt(e.mean) + attempt(e.std)
            best = min(best, (t, block, warps))
    kernels.EWM_CONFIG = best[1:]
    print(f"ewm: {best[1:]}, {best[0] * 1e3:.2f} ms", flush=True)
    return [best[1], best[2]]


def apply(path: str | Path) -> None:
    """Load a tuning file into torchrolling's kernel tables."""
    found = json.loads(Path(path).read_text())
    for group, table in found["rolling"].items():
        kernels.ROLLING_CONFIGS[group] = {int(k): (v[0], v[1]) for k, v in table.items()}
    kernels.QUANTILE_CONFIGS = {int(k): (v[0], v[1], v[2]) for k, v in found["quantile"].items()}
    kernels.EWM_CONFIG = (found["ewm"][0], found["ewm"][1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--length", type=int, default=20_000)
    args = parser.parse_args()
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.length, device="cuda")
    y = 0.5 * x + torch.randn_like(x)
    found = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "rolling": tune_rolling(x, y),
        "quantile": tune_quantile(x[: max(1, args.rows // 5)]),
        "ewm": tune_ewm(x),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tuning.json").write_text(json.dumps(found, indent=1))
    print(f"\nWrote {RESULTS / 'tuning.json'}")


if __name__ == "__main__":
    main()

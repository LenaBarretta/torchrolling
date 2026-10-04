"""Measure launch configurations of torchrolling's Triton kernels on this GPU.

    python bench/tune.py                          # every part; writes bench/results/tuning.json
    python bench/tune.py --parts direct,ewm       # only some parts
    python bench/bench.py --tuning bench/results/tuning.json

For every kernel and window size it tries tile sizes and warp counts (and where two kernels
can compute the same thing, both), keeps the fastest, and writes them in the shape of the
tables at the top of ``torchrolling/_triton.py``. ``apply`` loads such a file into those
tables, for bench.py; the best ones end up in the source. Parts not measured keep the
tables' current values.
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
PARTS = ["rolling", "direct", "quantile", "ewm"]
# Representative statistics per rolling group: their times are added up.
GROUPS = {"light": ["mean", "max"], "moments": ["std"], "heavy": ["corr", "kurt"]}
# The T4 run that filled the source tables preferred small tiles and 2-4 warps.
WARPS = [2, 4]
# Van Herk kernel. Sizes in between use the closest measured size below.
BLOCK_WS = [16, 64, 256, 1024, 4096]
TILES = [256, 512, 1024, 2048, 4096]
# Direct kernel, and the windows where it competes with van Herk.
DIRECT_WINDOWS = [4, 8, 16, 32, 64, 128]
DIRECT_TILES = [512, 1024, 2048]
# Quantile windows the select kernel is measured for (smaller ones are in the source).
QUANTILE_WINDOWS = [2048, 4096]
SELECT_BLOCKS = [256, 512, 1024]
EWM_BLOCKS = [512, 1024, 2048]
# A configuration slower than this (seconds) is not worth timing more than once.
SLOW = 1.0
START = time.perf_counter()


def log(message: str) -> None:
    print(f"[{time.perf_counter() - START:6.0f}s] {message}", flush=True)


def attempt(fn: Callable[[], object], repeats: int = 3) -> float:
    """Best time of ``fn``, or infinity when the configuration does not compile or run."""
    try:
        fn()  # compiles
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if times[-1] > SLOW:
                break
        return min(times)
    except Exception as e:  # out of resources, compiler limits, ...
        log(f"  skipped: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        return float("inf")


def group_time(group: str, x: torch.Tensor, y: torch.Tensor, window: int) -> float:
    total = 0.0
    for stat in GROUPS[group]:
        r = torchrolling.rolling(x, window)
        args = (y,) if stat == "corr" else ()
        total += attempt(lambda r=r, stat=stat, args=args: getattr(r, stat)(*args))
    return total


def tune_rolling(x: torch.Tensor, y: torch.Tensor) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    saved = dict(kernels.DIRECT_WINDOW)
    kernels.DIRECT_WINDOW = {group: 0 for group in GROUPS}  # van Herk everywhere
    for group in GROUPS:
        out[group] = {}
        for block_w in BLOCK_WS:
            window = max(1, block_w * 3 // 4)
            best = (float("inf"), 0, 0)
            for tile in [t for t in TILES if t >= block_w] or [block_w]:
                for warps in WARPS:
                    kernels.ROLLING_CONFIGS[group][block_w] = (tile, warps)
                    best = min(best, (group_time(group, x, y, window), tile, warps))
            kernels.ROLLING_CONFIGS[group][block_w] = best[1:]
            out[group][str(block_w)] = [best[1], best[2]]
            log(f"rolling {group} BLOCK_W={block_w}: {best[1:]}, {best[0] * 1e3:.2f} ms")
    kernels.DIRECT_WINDOW = saved
    return out


def tune_direct(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    """Configurations of the direct kernel, and up to which window it beats van Herk."""
    out: dict[str, Any] = {"configs": {}, "window": {}}
    for group in GROUPS:
        out["configs"][group] = {}
        largest = 0
        for window in DIRECT_WINDOWS:
            block_w = max(4, window)
            kernels.DIRECT_WINDOW[group] = 0
            van_herk = group_time(group, x, y, window)
            kernels.DIRECT_WINDOW[group] = window
            best = (float("inf"), 0, 0)
            for tile in [t for t in DIRECT_TILES if t >= block_w] or [block_w]:
                for warps in WARPS:
                    kernels.DIRECT_CONFIGS[group][block_w] = (tile, warps)
                    best = min(best, (group_time(group, x, y, window), tile, warps))
            kernels.DIRECT_CONFIGS[group][block_w] = best[1:]
            out["configs"][group][str(block_w)] = [best[1], best[2]]
            if best[0] < van_herk:
                largest = window
            log(
                f"direct {group} window={window}: {best[1:]}, {best[0] * 1e3:.2f} ms"
                f" (van Herk {van_herk * 1e3:.2f} ms)"
            )
        kernels.DIRECT_WINDOW[group] = largest
        out["window"][group] = largest
        log(f"direct {group}: up to window {largest}")
    return out


def tune_quantile(x: torch.Tensor) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for window in QUANTILE_WINDOWS:
        best: tuple[float, tuple[str, int, int]] = (float("inf"), ("select", 512, 4))
        for block in (b for b in SELECT_BLOCKS if b + window <= 8192):
            for warps in WARPS:
                kernels.QUANTILE_CONFIGS[window] = ("select", block, warps)
                t = attempt(lambda window=window: torchrolling.rolling(x, window).median())
                best = min(best, (t, ("select", block, warps)))
        kernels.QUANTILE_CONFIGS[window] = best[1]
        out[str(window)] = list(best[1])
        log(f"quantile window={window}: {best[1]}, {best[0] * 1e3:.2f} ms")
    return out


def tune_ewm(x: torch.Tensor) -> list[int]:
    best = (float("inf"), 1024, 4)
    for block in EWM_BLOCKS:
        for warps in WARPS:
            kernels.EWM_CONFIG = (block, warps)
            e = torchrolling.ewm(x, span=50)
            best = min(best, (attempt(e.mean) + attempt(e.std), block, warps))
    kernels.EWM_CONFIG = best[1:]
    log(f"ewm: {best[1:]}, {best[0] * 1e3:.2f} ms")
    return [best[1], best[2]]


def apply(path: str | Path) -> None:
    """Load a tuning file into torchrolling's kernel tables (only the parts it has)."""
    found = json.loads(Path(path).read_text())
    for group, table in found.get("rolling", {}).items():
        kernels.ROLLING_CONFIGS[group] = {int(k): (v[0], v[1]) for k, v in table.items()}
    if "direct" in found:
        for group, table in found["direct"]["configs"].items():
            kernels.DIRECT_CONFIGS[group] = {int(k): (v[0], v[1]) for k, v in table.items()}
        kernels.DIRECT_WINDOW = dict(found["direct"]["window"])
    for window, (kind, block, warps) in found.get("quantile", {}).items():
        kernels.QUANTILE_CONFIGS[int(window)] = (kind, block, warps)
    if "ewm" in found:
        kernels.EWM_CONFIG = (found["ewm"][0], found["ewm"][1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parts", default=",".join(PARTS), help=f"comma-separated: {PARTS}")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--length", type=int, default=20_000)
    args = parser.parse_args()
    parts = args.parts.split(",")
    torch.manual_seed(0)
    x = torch.randn(args.rows, args.length, device="cuda")
    y = 0.5 * x + torch.randn_like(x)
    found: dict[str, Any] = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    if "rolling" in parts:
        found["rolling"] = tune_rolling(x, y)
    if "direct" in parts:
        found["direct"] = tune_direct(x, y)
    if "quantile" in parts:
        found["quantile"] = tune_quantile(x[: max(1, args.rows // 5)])
    if "ewm" in parts:
        found["ewm"] = tune_ewm(x)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tuning.json").write_text(json.dumps(found, indent=1))
    log(f"wrote {RESULTS / 'tuning.json'}")


if __name__ == "__main__":
    main()

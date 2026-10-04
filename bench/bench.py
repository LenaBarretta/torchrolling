"""Benchmark torchrolling against pandas, polars, cuDF and a plain torch ``unfold``.

    python bench/bench.py            # full grid
    python bench/bench.py --quick    # small grid, to check that everything runs

Writes ``bench/results/<device>.json`` and ``bench/results/<device>.md``. Data is created
where each library wants it (GPU tensors for torch on CUDA, a DataFrame for pandas) and
only the rolling computation itself is timed: torchrolling is meant for series that already
live on the GPU. Libraries that are not installed are skipped. Data is float32. On CUDA,
torchrolling runs twice: with its default float32 accumulator and with
``acc_dtype=torch.float64``, which shows what the extra precision costs on that GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

import torchrolling

METHODS = ["mean", "std", "max", "median", "corr", "ewm_mean"]
FULL = {"shapes": [(100, 10_000), (1_000, 10_000), (10_000, 10_000)], "windows": [20, 200, 1000]}
QUICK = {"shapes": [(100, 2_000)], "windows": [10, 100]}
RESULTS = Path(__file__).parent / "results"
CPU_UNFOLD_BYTES = 4 * 2**30

Fn = Callable[[], object]
Make = Callable[[int, str], Fn]


def best_time(fn: Fn, sync: Callable[[], None], repeats: int) -> float:
    fn()  # warm-up; also compiles Triton kernels
    sync()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - start)
    return min(times)


def torch_impls(data: np.ndarray, other: np.ndarray, device: str) -> dict[str, Make]:
    x, y = torch.tensor(data, device=device), torch.tensor(other, device=device)
    reductions = {
        "mean": lambda t: t.mean(-1),
        "std": lambda t: t.std(-1),
        "max": lambda t: t.amax(-1),
        "median": lambda t: t.nanquantile(0.5, -1),
    }

    def corr(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u, v = u - u.mean(-1, keepdim=True), v - v.mean(-1, keepdim=True)
        return (u * v).sum(-1) / ((u * u).sum(-1) * (v * v).sum(-1)).sqrt()

    def unfold(w: int, m: str) -> Fn:
        if m == "ewm_mean":
            raise NotImplementedError("torch has no exponential moving average")
        # The reduction copies the unfolded view: window times the input size.
        if device == "cpu" and x.numel() * w * x.element_size() > CPU_UNFOLD_BYTES:
            raise MemoryError(f"needs {x.numel() * w * x.element_size() / 2**30:.0f} GB")
        if m == "corr":
            return lambda: corr(x.unfold(-1, w, 1), y.unfold(-1, w, 1))
        return lambda: reductions[m](x.unfold(-1, w, 1))

    def ours(acc: torch.dtype | None) -> Make:
        def make(w: int, m: str) -> Fn:
            if m == "ewm_mean":
                return lambda: torchrolling.ewm(x, span=w, acc_dtype=acc).mean()
            if m == "corr":
                return lambda: torchrolling.rolling(x, w, acc_dtype=acc).corr(y)
            return lambda: getattr(torchrolling.rolling(x, w, acc_dtype=acc), m)()

        return make

    impls = {f"torchrolling ({device})": ours(None)}
    if device == "cuda":
        impls[f"torchrolling fp64 acc ({device})"] = ours(torch.float64)
    impls[f"torch unfold ({device})"] = unfold
    return impls


def pandas_impls(data: np.ndarray, other: np.ndarray) -> dict[str, Make]:
    import pandas as pd

    frame, frame2 = pd.DataFrame(data.T), pd.DataFrame(other.T)

    def make(w: int, m: str) -> Fn:
        if m == "ewm_mean":
            return lambda: frame.ewm(span=w).mean()
        if m == "corr":
            return lambda: frame.rolling(w).corr(frame2)
        return lambda: getattr(frame.rolling(w), m)()

    return {"pandas": make}


def polars_impls(data: np.ndarray) -> dict[str, Make]:
    try:
        import polars as pl
    except ImportError:
        return {}
    frame = pl.DataFrame(data.T)

    def make(w: int, m: str) -> Fn:
        if m == "ewm_mean":
            return lambda: frame.select(pl.all().ewm_mean(span=w))
        if m == "corr":
            raise NotImplementedError("polars has no column-wise rolling corr")
        return lambda: frame.select(getattr(pl.all(), f"rolling_{m}")(w))

    return {"polars": make}


def cudf_impls(data: np.ndarray, other: np.ndarray) -> dict[str, Make]:
    try:
        import cudf
    except ImportError:
        return {}
    frame, frame2 = cudf.DataFrame(data.T), cudf.DataFrame(other.T)

    def make(w: int, m: str) -> Fn:
        if m == "ewm_mean":
            return lambda: frame.ewm(span=w).mean()
        if m == "corr":
            return lambda: frame.rolling(w).corr(frame2)
        return lambda: getattr(frame.rolling(w), m)()

    return {"cuDF": make}


def run(grid: dict[str, Any], repeats: int, max_cpu_elements: int) -> list[dict[str, Any]]:
    cuda = torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    rows = []
    for shape in grid["shapes"]:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(shape).astype(np.float32)
        other = (0.5 * data + rng.standard_normal(shape)).astype(np.float32)
        impls = torch_impls(data, other, "cuda") if cuda else {}
        if data.size <= max_cpu_elements:
            impls |= (
                torch_impls(data, other, "cpu") | pandas_impls(data, other) | polars_impls(data)
            )
        if cuda:
            impls |= cudf_impls(data, other)
        for window in grid["windows"]:
            for method in METHODS:
                for name, make in impls.items():
                    row = {"shape": list(shape), "window": window, "method": method, "impl": name}
                    try:
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                        row["seconds"] = best_time(make(window, method), sync, repeats)
                        if cuda and "cuda" in name:
                            row["peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
                    except torch.cuda.OutOfMemoryError:
                        row["error"] = "out of memory"
                    except MemoryError as e:
                        row["error"] = f"skipped, {e}"
                    except NotImplementedError as e:  # a RuntimeError, so first
                        row["error"] = f"not supported ({e})"
                    except RuntimeError as e:
                        row["error"] = f"failed ({str(e).splitlines()[0][:60]})"
                    except (AttributeError, TypeError) as e:
                        row["error"] = f"not supported ({type(e).__name__})"
                    if cuda:
                        torch.cuda.empty_cache()
                    rows.append(row)
                    print(row, flush=True)
    return rows


def markdown(rows: list[dict[str, Any]], device: str) -> str:
    impls = list(dict.fromkeys(r["impl"] for r in rows))
    lines = [f"Device: {device}. Best of several runs, in milliseconds (lower is better).", ""]
    for method in METHODS:
        lines += [f"### {method}", ""]
        lines.append("| series x length | window | " + " | ".join(impls) + " |")
        lines.append("| --- | --- |" + " --- |" * len(impls))
        cases = dict.fromkeys((tuple(r["shape"]), r["window"]) for r in rows)
        for shape, window in cases:
            cells = []
            for impl in impls:
                match = [
                    r
                    for r in rows
                    if r["impl"] == impl
                    and r["method"] == method
                    and tuple(r["shape"]) == shape
                    and r["window"] == window
                ]
                if not match:
                    cells.append("-")
                elif "seconds" in match[0]:
                    cells.append(f"{match[0]['seconds'] * 1000:.1f}")
                else:
                    cells.append(match[0]["error"])
            lines.append(f"| {shape[0]:,} x {shape[1]:,} | {window} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small grid")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-cpu-elements",
        type=int,
        default=10_000_000,
        help="skip CPU libraries on bigger inputs (they get slow)",
    )
    args = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    else:
        device = (
            f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"
        )
    rows = run(QUICK if args.quick else FULL, args.repeats, args.max_cpu_elements)
    RESULTS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", device.lower()).strip("-")
    meta = {"device": device, "torch": torch.__version__, "torchrolling": torchrolling.__version__}
    (RESULTS / f"{slug}.json").write_text(json.dumps({**meta, "rows": rows}, indent=1))
    (RESULTS / f"{slug}.md").write_text(markdown(rows, device))
    print(f"\nWrote {RESULTS / slug}.json and .md")


if __name__ == "__main__":
    main()

import os
from collections.abc import Iterator
from unittest import mock

import pytest
import torch
from hypothesis import HealthCheck, settings

# Without a GPU, run the Triton kernels on CPU through Triton's interpreter.
# This has to happen before torchrolling (and so triton.jit) is imported.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from torchrolling import _common

# The backend fixture only patches module attributes, the same for every example.
settings.register_profile("default", suppress_health_check=[HealthCheck.function_scoped_fixture])
settings.load_profile("default")

BACKENDS = ["torch", "triton", "triton-small"]


def _never(*tensors: torch.Tensor, window: int = 0, limit: str = "") -> bool:
    return False


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[str]:
    """Run a test on the torch code, or on the Triton kernels (where Triton is installed).

    "triton-small" shrinks the kernels' tiles and chunks so that small inputs still cross
    tile boundaries, and sends every quantile window to the select kernel.
    """
    name: str = request.param
    if name == "torch":
        # A plain function rather than a Mock, so that torch.compile can trace through it.
        with mock.patch.object(_common, "use_triton", _never):
            yield name
        return
    if _common._triton is None:  # type: ignore[attr-defined]
        pytest.skip("needs Triton")
    if name == "triton":  # pragma: no cover - needs Triton
        if os.environ.get("TRITON_INTERPRET") == "1" and getattr(
            request.function, "is_hypothesis_test", False
        ):
            # Tile sizes only change speed. In the interpreter, full-size tiles make every
            # tiny example slow; triton-small runs the same code across more tile edges.
            pytest.skip("full-size tiles: covered by triton-small and the long-series tests")
        yield name
        return
    from torchrolling import _triton  # pragma: no cover - needs Triton

    tiny = {group: (32, 4) for group in _triton.ROLLING_DEFAULT}  # pragma: no cover
    with mock.patch.multiple(  # pragma: no cover - needs Triton
        _triton,
        ROLLING_CONFIGS={group: {} for group in tiny},
        ROLLING_DEFAULT=tiny,
        QUANTILE_CONFIGS={},
        SORT_WINDOW=1,
        EWM_CONFIG=(16, 4),
    ):
        yield name

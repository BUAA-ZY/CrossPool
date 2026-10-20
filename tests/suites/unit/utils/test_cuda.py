from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from cuda.bindings import driver, runtime  # ty: ignore[unresolved-import]

from xpool.utils import cuda


def test_runtime_check_preserves_status_and_operation() -> None:
    assert cuda.check_runtime(runtime.cudaError_t.cudaSuccess, "query") is None
    status = runtime.cudaError_t.cudaErrorMemoryAllocation
    with pytest.raises(cuda.CudaRuntimeError) as caught:
        cuda.check_runtime(status, "cudaMalloc")
    assert caught.value.error_code is status
    assert caught.value.operation == "cudaMalloc"
    assert not cuda.is_terminal_device_error(caught.value)


def test_driver_check_preserves_status_and_operation() -> None:
    assert cuda.check_driver(driver.CUresult.CUDA_SUCCESS, "query") is None
    status = driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY
    with pytest.raises(cuda.CudaDriverError) as caught:
        cuda.check_driver(status, "cuMemCreate")
    assert caught.value.error_code is status
    assert caught.value.operation == "cuMemCreate"
    assert not cuda.is_terminal_device_error(caught.value)


@pytest.mark.parametrize(
    "failure",
    (
        cuda.CudaRuntimeError(runtime.cudaError_t.cudaErrorIllegalAddress, "cudaMemGetInfo"),
        cuda.CudaDriverError(driver.CUresult.CUDA_ERROR_LAUNCH_FAILED, "cuMemMap"),
    ),
)
def test_terminal_status_survives_exception_wrapping(failure: RuntimeError) -> None:
    wrapped = RuntimeError("measurement failed")
    wrapped.__cause__ = failure
    assert cuda.is_terminal_device_error(failure)
    assert cuda.is_terminal_device_error(wrapped)


def test_terminal_device_error_requires_retained_numeric_evidence() -> None:
    failure = torch.AcceleratorError("device execution failed")
    setattr(failure, "error_code", int(runtime.cudaError_t.cudaErrorLaunchFailure))
    wrapped = RuntimeError("runtime failed")
    wrapped.__cause__ = failure

    assert cuda.is_terminal_device_error(failure)
    assert cuda.is_terminal_device_error(wrapped)
    wrapped.__cause__ = None
    wrapped.__context__ = failure
    assert cuda.is_terminal_device_error(wrapped)
    setattr(failure, "error_code", int(runtime.cudaError_t.cudaErrorInvalidValue))
    assert not cuda.is_terminal_device_error(wrapped)
    assert not cuda.is_terminal_device_error(torch.AcceleratorError("unspecified launch failure"))
    assert not cuda.is_terminal_device_error(RuntimeError("CUDA error 719"))


@pytest.mark.parametrize("first_status", (runtime.cudaError_t.cudaSuccess, runtime.cudaError_t.cudaErrorIllegalAddress))
def test_version_queries_check_each_result_before_proceeding(
    monkeypatch: pytest.MonkeyPatch, first_status: runtime.cudaError_t
) -> None:
    queried: list[str] = []

    def driver_version() -> tuple[runtime.cudaError_t, int]:
        queried.append("driver")
        return first_status, 13030

    def runtime_version() -> tuple[runtime.cudaError_t, int]:
        queried.append("runtime")
        return runtime.cudaError_t.cudaSuccess, 13031

    monkeypatch.setattr(
        cuda,
        "runtime",
        SimpleNamespace(
            cudaError_t=runtime.cudaError_t,
            cudaDriverGetVersion=driver_version,
            cudaRuntimeGetVersion=runtime_version,
        ),
    )
    if first_status == runtime.cudaError_t.cudaSuccess:
        assert cuda.cuda_versions() == (13030, 13031)
        assert queried == ["driver", "runtime"]
    else:
        with pytest.raises(cuda.CudaRuntimeError) as caught:
            cuda.cuda_versions()
        assert caught.value.error_code is first_status
        assert caught.value.operation == "cudaDriverGetVersion"
        assert queried == ["driver"]

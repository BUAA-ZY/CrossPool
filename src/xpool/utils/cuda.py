"""CUDA result checks and retained terminal-error classification."""

from __future__ import annotations

import torch

# cuda-bindings exposes these binary modules without Python type stubs.
from cuda.bindings import driver, runtime  # ty: ignore[unresolved-import]

__all__ = [
    "CudaDriverError",
    "CudaRuntimeError",
    "check_driver",
    "check_runtime",
    "cuda_versions",
    "is_terminal_device_error",
]


class CudaRuntimeError(RuntimeError):
    """Retain a Runtime API failure's original status and operation."""

    def __init__(self, error_code: runtime.cudaError_t, operation: str) -> None:
        self.error_code = error_code
        self.operation = operation
        super().__init__(f"{operation}: {error_code.name}")


class CudaDriverError(RuntimeError):
    """Retain a Driver API failure's original status and operation."""

    def __init__(self, error_code: driver.CUresult, operation: str) -> None:
        self.error_code = error_code
        self.operation = operation
        super().__init__(f"{operation}: {error_code.name}")


def check_runtime(error_code: runtime.cudaError_t, operation: str) -> None:
    """Raise CudaRuntimeError for a returned non-success Runtime status."""

    if error_code != runtime.cudaError_t.cudaSuccess:
        raise CudaRuntimeError(error_code, operation)


def check_driver(error_code: driver.CUresult, operation: str) -> None:
    """Raise CudaDriverError for a returned non-success Driver status."""

    if error_code != driver.CUresult.CUDA_SUCCESS:
        raise CudaDriverError(error_code, operation)


def cuda_versions() -> tuple[int, int]:
    """Read driver and Runtime versions without creating a CUDA context.

    Both queries use the Runtime API and preserve a failed query's status.
    """

    status, driver_version = runtime.cudaDriverGetVersion()
    check_runtime(status, "cudaDriverGetVersion")
    status, runtime_version = runtime.cudaRuntimeGetVersion()
    check_runtime(status, "cudaRuntimeGetVersion")
    return int(driver_version), int(runtime_version)


def is_terminal_device_error(error: BaseException) -> bool:
    """Inspect retained CUDA results that explicitly require process exit.

    Follow the original exception's standard cause/context chain. Torch's
    native translator supplies numeric metadata dynamically; missing metadata
    and ordinary failures supply no terminal-exit evidence. Runtime and Driver
    exceptions retain their own status domains.

    This predicate performs no device or MPS query. The calling lifecycle owner
    decides process-exit and resource policy; a terminal result establishes no
    peer or complete-domain retirement.
    """

    runtime_codes = (
        runtime.cudaError_t.cudaErrorContained,
        runtime.cudaError_t.cudaErrorIllegalAddress,
        runtime.cudaError_t.cudaErrorLaunchTimeout,
        runtime.cudaError_t.cudaErrorAssert,
        runtime.cudaError_t.cudaErrorHardwareStackError,
        runtime.cudaError_t.cudaErrorIllegalInstruction,
        runtime.cudaError_t.cudaErrorMisalignedAddress,
        runtime.cudaError_t.cudaErrorInvalidAddressSpace,
        runtime.cudaError_t.cudaErrorInvalidPc,
        runtime.cudaError_t.cudaErrorLaunchFailure,
        runtime.cudaError_t.cudaErrorMpsClientTerminated,
    )
    driver_codes = (
        driver.CUresult.CUDA_ERROR_CONTAINED,
        driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS,
        driver.CUresult.CUDA_ERROR_LAUNCH_TIMEOUT,
        driver.CUresult.CUDA_ERROR_ASSERT,
        driver.CUresult.CUDA_ERROR_HARDWARE_STACK_ERROR,
        driver.CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION,
        driver.CUresult.CUDA_ERROR_MISALIGNED_ADDRESS,
        driver.CUresult.CUDA_ERROR_INVALID_ADDRESS_SPACE,
        driver.CUresult.CUDA_ERROR_INVALID_PC,
        driver.CUresult.CUDA_ERROR_LAUNCH_FAILED,
        driver.CUresult.CUDA_ERROR_MPS_CLIENT_TERMINATED,
    )
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, CudaRuntimeError):
            if current.error_code in runtime_codes:
                return True
        elif isinstance(current, CudaDriverError):
            if current.error_code in driver_codes:
                return True
        elif isinstance(current, torch.AcceleratorError) and getattr(current, "error_code", None) in runtime_codes:
            return True
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return False

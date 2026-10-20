from __future__ import annotations

import pytest
import torch
from cuda.bindings import driver, runtime  # ty: ignore[unresolved-import]
from sglang.srt.layers.radix_attention import RadixAttention

import xpool.integrations.sglang.kv.vmm
import xtest
from xpool.integrations.sglang.kv.pool import ElasticMHATokenToKVPool, ElasticMLATokenToKVPool
from xpool.integrations.sglang.kv.vmm import KvVmmBacking
from xpool.service.wire import KvCapacityPartitionProfile
from xpool.utils.cuda import CudaDriverError, CudaRuntimeError, check_driver, is_terminal_device_error
from xtest.harness.support.sglang.runtime import published_sglang_config


@pytest.fixture
def backing() -> KvVmmBacking:
    backing = object.__new__(KvVmmBacking)
    backing.capacity_profile = KvCapacityPartitionProfile(
        bundle_bytes=2048,
        bundle_capacity=2,
        floor_bundles=1,
        token_capacity=1,
        mapping_granularity_bytes=2048,
        row_bytes=2048,
        tokens_per_row=1,
        token_page_size=1,
    )
    backing.base = 4096
    backing.allocation_properties = object()
    backing.access_descriptors = [object()]
    backing.reserved_bytes = 4096
    backing.backed_bundles = 0
    backing.raw_storage = None
    backing.tensors = []
    return backing


@pytest.mark.parametrize("diagnostic", ("available", "unavailable", "terminal"))
def test_bundle_allocation_failure_reports_physical_state(
    backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch, diagnostic: str
) -> None:
    backing.backed_bundles = 1
    diagnostic_error = (
        CudaRuntimeError(runtime.cudaError_t.cudaErrorIllegalAddress, "cudaMemGetInfo")
        if diagnostic == "terminal"
        else RuntimeError("memory query unavailable")
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemCreate",
        lambda *args: (driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY, 0),
    )

    def memory_info(device: torch.device | None) -> tuple[int, int]:
        if diagnostic != "available":
            raise diagnostic_error
        return 1024, 4096

    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.torch.cuda, "mem_get_info", memory_info)

    with pytest.raises(RuntimeError) as raised:
        backing.resize(2)
    assert backing.backed_bundles == 1
    if diagnostic == "terminal":
        assert raised.value is diagnostic_error
        assert is_terminal_device_error(raised.value)
        allocation_error = raised.value.__cause__
    else:
        allocation_error = raised.value
        assert not is_terminal_device_error(allocation_error)
        assert allocation_error.__notes__[-1] == (
            "free_bytes=1024 total_bytes=4096"
            if diagnostic == "available"
            else "CUDA memory query failed: memory query unavailable"
        )
    assert isinstance(allocation_error, CudaDriverError)
    assert allocation_error.error_code is driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY
    assert allocation_error.operation == "cuMemCreate(xpool kv bundle)"
    assert allocation_error.__notes__[0] == "target_bundles=2 backed_bundles=1 bundle_bytes=2048"


def test_partial_bundle_mapping_retains_actual_progress_for_close(
    backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    unmapped: list[int] = []
    released: list[int] = []
    freed: list[tuple[int, int]] = []
    next_handle = 0

    def create(byte_count: int, properties: object, flags: int) -> tuple[driver.CUresult, int]:
        nonlocal next_handle
        assert byte_count == backing.capacity_profile.bundle_bytes
        assert flags == 0
        handle = next_handle
        next_handle += 1
        return driver.CUresult.CUDA_SUCCESS, handle

    def map_bundle(address: int, byte_count: int, offset: int, handle: int, flags: int) -> tuple[driver.CUresult]:
        assert byte_count == backing.capacity_profile.bundle_bytes
        assert offset == flags == 0
        assert handle in (0, 1)
        return (driver.CUresult.CUDA_SUCCESS,)

    def set_access(address: int, byte_count: int, descriptors: list[object], count: int) -> tuple[driver.CUresult]:
        assert byte_count == backing.capacity_profile.bundle_bytes
        assert count == 1
        return (
            driver.CUresult.CUDA_ERROR_INVALID_VALUE
            if address == backing.base + backing.capacity_profile.bundle_bytes
            else driver.CUresult.CUDA_SUCCESS,
        )

    def unmap(address: int, byte_count: int) -> tuple[driver.CUresult]:
        unmapped.append(address)
        return (driver.CUresult.CUDA_SUCCESS,)

    def release(handle: int) -> tuple[driver.CUresult]:
        released.append(handle)
        return (driver.CUresult.CUDA_SUCCESS,)

    def free(address: int, byte_count: int) -> tuple[driver.CUresult]:
        freed.append((address, byte_count))
        return (driver.CUresult.CUDA_SUCCESS,)

    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemCreate", create)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemMap", map_bundle)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemSetAccess", set_access)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemUnmap", unmap)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemRelease", release)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.driver, "cuMemAddressFree", free)

    with pytest.raises(RuntimeError, match="cuMemSetAccess"):
        backing.resize(2)

    assert backing.backed_bundles == 1
    assert unmapped == [backing.base + backing.capacity_profile.bundle_bytes]
    assert released == [0, 1]

    backing.close()

    assert backing.backed_bundles == 0
    assert backing.base == 0
    assert unmapped == [4096 + 2048, 4096]
    assert freed == [(4096, 4096)]


def test_terminal_allocation_failure_skips_memory_diagnostic(
    backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemCreate",
        lambda *args: (driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS, 0),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.torch.cuda,
        "mem_get_info",
        lambda *args: pytest.fail("memory query after terminal result"),
    )
    with pytest.raises(CudaDriverError) as caught:
        backing.resize(1)
    assert caught.value.error_code is driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS


def test_terminal_mapping_failure_skips_device_rollback(backing: KvVmmBacking, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver, "cuMemCreate", lambda *args: (driver.CUresult.CUDA_SUCCESS, 1)
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver, "cuMemMap", lambda *args: (driver.CUresult.CUDA_SUCCESS,)
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemSetAccess",
        lambda *args: (driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS,),
    )
    for name in ("cuMemUnmap", "cuMemRelease"):
        monkeypatch.setattr(
            xpool.integrations.sglang.kv.vmm.driver,
            name,
            lambda *args: pytest.fail("device rollback after terminal result"),
        )
    with pytest.raises(CudaDriverError) as caught:
        backing.resize(1)
    assert caught.value.operation == "cuMemSetAccess(xpool kv bundle)"
    assert backing.backed_bundles == 0


@pytest.mark.parametrize("terminal", (False, True))
def test_constructor_cleanup_depends_on_resource_usability(monkeypatch: pytest.MonkeyPatch, terminal: bool) -> None:
    closed: list[int] = []
    terminal_seen = False
    status = driver.CUresult.CUDA_ERROR_ILLEGAL_ADDRESS if terminal else driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY

    def resize(self: KvVmmBacking, bundle_count: int) -> None:
        nonlocal terminal_seen
        terminal_seen = terminal
        raise CudaDriverError(status, "cuMemCreate")

    def exchange_device(index: int) -> int:
        if terminal_seen:
            pytest.fail("device selection after terminal result")
        return index

    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemGetAllocationGranularity",
        lambda *args: (driver.CUresult.CUDA_SUCCESS, 2048),
    )
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm.driver,
        "cuMemAddressReserve",
        lambda *args: (driver.CUresult.CUDA_SUCCESS, 4096),
    )
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.torch.cuda, "_exchange_device", exchange_device)
    monkeypatch.setattr(xpool.integrations.sglang.kv.vmm.torch.cuda, "_maybe_exchange_device", exchange_device)
    monkeypatch.setattr(
        xpool.integrations.sglang.kv.vmm,
        "tensor_from_pointer",
        lambda pointer, byte_count, **kwargs: torch.empty(byte_count, dtype=torch.uint8),
    )
    monkeypatch.setattr(KvVmmBacking, "resize", resize)
    monkeypatch.setattr(KvVmmBacking, "close", lambda self: closed.append(self.base))
    descriptor = xpool.integrations.sglang.kv.vmm.KvBufferDesc("kv", (2, 1), row_bytes=1, tokens_per_row=1)
    with pytest.raises(CudaDriverError) as caught:
        KvVmmBacking(
            device=torch.device("cuda:0"),
            storage_dtype=torch.uint8,
            token_page_size=1,
            descriptors=(descriptor,),
            buffers_per_layer=1,
        )
    assert caught.value.error_code is status
    assert closed == ([] if terminal else [4096])


@xtest.requirements(device_count=1)
def test_mha_pool_preserves_views_across_reversible_bundle_mapping() -> None:
    pool = ElasticMHATokenToKVPool(2048, 1, torch.float16, 8, 64, 2, "cuda", False)
    try:
        properties = pool.backing.allocation_properties
        assert properties.requestedHandleTypes == driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        status, granularity = driver.cuMemGetAllocationGranularity(
            properties, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        check_driver(status, "cuMemGetAllocationGranularity")
        assert pool.backing.capacity_profile.mapping_granularity_bytes == granularity
        keys_by_layer = pool.k_buffer
        values_by_layer = pool.v_buffer
        pointers = tuple(tensor.data_ptr() for tensor in (*keys_by_layer, *values_by_layer))
        pool.backing.resize(pool.backing.capacity_profile.bundle_capacity)

        locations = torch.tensor([1, 2048], device="cuda")
        keys = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
        values = torch.randn_like(keys)
        layer = RadixAttention(num_heads=8, head_dim=64, scaling=64**-0.5, num_kv_heads=8, layer_id=0)
        pool.set_kv_buffer(layer, locations, keys, values)
        torch.cuda.synchronize()
        assert torch.equal(keys_by_layer[0][locations], keys)
        assert torch.equal(values_by_layer[0][locations], values)

        graph_input = keys_by_layer[0][1]
        graph_output = torch.empty_like(graph_input)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.add(graph_input, 1, out=graph_output)
        pool.backing.resize(pool.backing.capacity_profile.floor_bundles)
        pool.backing.resize(pool.backing.capacity_profile.bundle_capacity)
        graph.replay()
        torch.cuda.synchronize()

        assert tuple(tensor.data_ptr() for tensor in (*keys_by_layer, *values_by_layer)) == pointers
        assert torch.equal(graph_output, graph_input + 1)
    finally:
        pool.close()


@xtest.requirements(device_count=1)
@pytest.mark.usefixtures(published_sglang_config.__name__)
def test_mla_pool_writes_compound_strided_views() -> None:
    pool = ElasticMLATokenToKVPool(2048, 1, torch.float16, 512, 64, 2, "cuda", False)
    try:
        locations = torch.tensor([1, 2], device="cuda")
        hidden_states = torch.randn(2, 1, 576, device="cuda", dtype=torch.float16)
        pool.set_kv_buffer(
            RadixAttention(num_heads=1, head_dim=576, scaling=576**-0.5, num_kv_heads=1, layer_id=0),
            locations,
            hidden_states,
            torch.empty(0, device="cuda"),
        )
        torch.cuda.synchronize()
        assert torch.equal(pool.kv_buffer[0][locations], hidden_states)
    finally:
        pool.close()

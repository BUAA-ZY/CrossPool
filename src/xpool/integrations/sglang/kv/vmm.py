"""Reversible compound CUDA VMM backing for one SGLang KV pool."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from cuda.bindings import driver  # ty: ignore[unresolved-import]
from sglang.srt.mem_cache.memory_pool import KvBufferDesc
from sglang.srt.utils.cuda_vmm_utils import (
    make_device_allocation_prop,
    make_rw_access_desc,
    tensor_from_pointer,
)

from xpool.service.wire import KvCapacityPartitionProfile
from xpool.utils import align_up
from xpool.utils.cuda import check_driver, is_terminal_device_error


class KvVmmBacking:
    """Own one stable compound virtual range and its mapped bundle prefix."""

    def __init__(
        self,
        *,
        device: torch.device,
        storage_dtype: torch.dtype,
        token_page_size: int,
        descriptors: Sequence[KvBufferDesc],
        buffers_per_layer: int,
    ) -> None:
        """Reserve the compound range, expose views, and map the bootstrap floor.

        The calling Instance Rank must have selected ``device`` as its current device.
        """

        if device.type != "cuda":
            raise ValueError("xpool elastic kv backing requires a device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if token_page_size <= 0 or buffers_per_layer <= 0 or not descriptors:
            raise ValueError("xpool elastic kv backing geometry must be positive")
        if len(descriptors) % buffers_per_layer != 0:
            raise ValueError("xpool elastic kv descriptors do not form complete layers")

        first = descriptors[0]
        if not first.shape or first.shape[0] <= 0:
            raise ValueError("xpool elastic kv descriptors require a positive row count")
        geometry = (first.shape, first.row_bytes, first.tokens_per_row)
        if any(
            (descriptor.shape, descriptor.row_bytes, descriptor.tokens_per_row) != geometry
            for descriptor in descriptors
        ):
            raise ValueError("xpool elastic kv compound components require identical geometry")

        rows = int(first.shape[0])
        row_shape = tuple(int(dimension) for dimension in first.shape[1:])
        row_bytes = int(first.row_bytes)
        if math.prod(row_shape) * storage_dtype.itemsize != row_bytes:
            raise ValueError("xpool elastic kv descriptor row bytes disagree with its shape and dtype")

        token_page_size = int(token_page_size)
        allocation_properties = make_device_allocation_prop(device.index, handle_types=None)
        status, mapping_granularity_bytes = driver.cuMemGetAllocationGranularity(
            allocation_properties,
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
        check_driver(status, "cuMemGetAllocationGranularity(xpool kv)")
        layer_count = len(descriptors) // buffers_per_layer
        tokens_per_row = int(first.tokens_per_row)
        token_capacity = rows * tokens_per_row - token_page_size
        if token_capacity <= 0:
            raise ValueError("xpool elastic kv reservation does not contain a usable token page")

        component_span_bytes = rows * row_bytes
        bundle_capacity = align_up(component_span_bytes, mapping_granularity_bytes) // mapping_granularity_bytes
        bundle_bytes = mapping_granularity_bytes * len(descriptors)
        bootstrap_rows = (token_page_size + tokens_per_row - 1) // tokens_per_row
        floor_bundles = max(
            1,
            (bootstrap_rows * row_bytes + mapping_granularity_bytes - 1) // mapping_granularity_bytes,
        )
        self.capacity_profile = KvCapacityPartitionProfile(
            bundle_bytes=bundle_bytes,
            bundle_capacity=bundle_capacity,
            floor_bundles=floor_bundles,
            token_capacity=token_capacity,
            mapping_granularity_bytes=mapping_granularity_bytes,
            row_bytes=row_bytes,
            tokens_per_row=tokens_per_row,
            token_page_size=token_page_size,
        )
        self.reserved_bytes = bundle_capacity * bundle_bytes
        self.base = 0
        self.backed_bundles = 0
        self.raw_storage: torch.Tensor | None = None
        self.tensors: list[torch.Tensor] = []

        self.access_descriptors = [make_rw_access_desc(device.index)]
        status, base = driver.cuMemAddressReserve(self.reserved_bytes, mapping_granularity_bytes, 0, 0)
        check_driver(status, "cuMemAddressReserve(xpool kv)")
        self.base = int(base)
        try:
            self.raw_storage = tensor_from_pointer(
                self.base,
                self.reserved_bytes,
                device_id=device.index,
            )
            logical_elements = rows * layer_count * buffers_per_layer * math.prod(row_shape)
            logical = (
                self.raw_storage[: logical_elements * storage_dtype.itemsize]
                .view(storage_dtype)
                .view(
                    rows,
                    layer_count,
                    buffers_per_layer,
                    *row_shape,
                )
            )
            self.tensors = [
                logical[:, layer, buffer, ...] for buffer in range(buffers_per_layer) for layer in range(layer_count)
            ]
            self.allocation_properties = allocation_properties
            self.resize(floor_bundles)
        except BaseException as error:
            if not is_terminal_device_error(error):
                self.close()
            raise

    def usable_tokens(self, bundle_count: int) -> int:
        """Return the page-aligned allocator capacity safely covered by a bundle prefix."""

        return self.capacity_profile.usable_tokens(bundle_count)

    def required_bundles(self, token_count: int) -> int:
        """Return the smallest compound-bundle count covering ``token_count`` tokens."""

        if token_count < 0:
            raise ValueError("xpool elastic kv token demand must be nonnegative")
        profile = self.capacity_profile
        required_rows = math.ceil((token_count + profile.token_page_size) / profile.tokens_per_row)
        return math.ceil(required_rows * profile.row_bytes / profile.mapping_granularity_bytes)

    def resize(self, bundle_count: int) -> None:
        """Map or unmap complete tail bundles until the physical prefix reaches ``bundle_count``."""

        profile = self.capacity_profile
        if not profile.floor_bundles <= bundle_count <= profile.bundle_capacity:
            raise ValueError("xpool elastic kv bundle count is outside its service range")
        if self.base == 0:
            raise RuntimeError("xpool elastic kv backing is closed")

        while self.backed_bundles < bundle_count:
            address = self.base + self.backed_bundles * profile.bundle_bytes
            try:
                status, handle = driver.cuMemCreate(profile.bundle_bytes, self.allocation_properties, 0)
                check_driver(status, "cuMemCreate(xpool kv bundle)")
            except RuntimeError as error:
                error.add_note(
                    f"target_bundles={bundle_count} backed_bundles={self.backed_bundles} "
                    f"bundle_bytes={profile.bundle_bytes}"
                )
                if is_terminal_device_error(error):
                    raise
                device = None if self.raw_storage is None else self.raw_storage.device
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                except RuntimeError as diagnostic_error:
                    if is_terminal_device_error(diagnostic_error):
                        raise diagnostic_error from error
                    error.add_note(f"CUDA memory query failed: {diagnostic_error}")
                else:
                    error.add_note(f"free_bytes={free_bytes} total_bytes={total_bytes}")
                raise
            mapped = False
            try:
                (status,) = driver.cuMemMap(address, profile.bundle_bytes, 0, handle, 0)
                check_driver(status, "cuMemMap(xpool kv bundle)")
                mapped = True
                (status,) = driver.cuMemSetAccess(
                    address,
                    profile.bundle_bytes,
                    self.access_descriptors,
                    len(self.access_descriptors),
                )
                check_driver(status, "cuMemSetAccess(xpool kv bundle)")
                (status,) = driver.cuMemRelease(handle)
                check_driver(status, "cuMemRelease(xpool kv bundle)")
                handle = None
            except BaseException as error:
                if is_terminal_device_error(error):
                    raise
                if mapped:
                    (status,) = driver.cuMemUnmap(address, profile.bundle_bytes)
                    check_driver(status, "cuMemUnmap(xpool kv rollback)")
                if handle is not None:
                    (status,) = driver.cuMemRelease(handle)
                    check_driver(status, "cuMemRelease(xpool kv rollback)")
                raise
            self.backed_bundles += 1

        while self.backed_bundles > bundle_count:
            address = self.base + (self.backed_bundles - 1) * profile.bundle_bytes
            (status,) = driver.cuMemUnmap(address, profile.bundle_bytes)
            check_driver(status, "cuMemUnmap(xpool kv bundle)")
            self.backed_bundles -= 1

    def close(self) -> None:
        """Drop every Tensor view, unmap the actual prefix, and release the virtual range."""

        if self.base == 0:
            return
        self.tensors.clear()
        self.raw_storage = None
        bundle_bytes = self.capacity_profile.bundle_bytes
        while self.backed_bundles:
            address = self.base + (self.backed_bundles - 1) * bundle_bytes
            (status,) = driver.cuMemUnmap(address, bundle_bytes)
            check_driver(status, "cuMemUnmap(xpool kv close)")
            self.backed_bundles -= 1
        (status,) = driver.cuMemAddressFree(self.base, self.reserved_bytes)
        check_driver(status, "cuMemAddressFree(xpool kv)")
        self.base = 0

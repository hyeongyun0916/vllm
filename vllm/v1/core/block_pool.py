# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable, Sequence
from typing import Any

from vllm import envs
from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_block_hashes,
)
from vllm.v1.core.priority_eviction_queue import PriorityEvictionQueue
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def contain(self, key: BlockHashWithGroupId, block_id: int) -> bool:
        """
        Checks whether the key maps to the given block ID.
        """
        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        if isinstance(blocks, dict):
            return block_id in blocks
        self._unexpected_blocks_type(blocks)
        return False

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

        # Sidecar storage for priority-based KV-cache eviction (empty until a
        # retention directive is applied).
        self.priority_eviction_queue = PriorityEvictionQueue()
        # Fail-safe budget: the protected pool may never grow large enough to
        # squeeze the unprotected working set into thrash. Protections beyond
        # the budget degrade to plain LRU instead of pinning.
        self.retention_budget_blocks = int(
            num_gpu_blocks * envs.VLLM_RETENTION_BUDGET_FRAC
        )

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
            block_mask: Optional mask aligned with
                ``blocks[num_cached_blocks:num_full_blocks]``. When provided,
                blocks where the mask is False are skipped (treated like null
                blocks). Used by groups whose ``find_longest_cache_hit`` only
                consults a subset of blocks (e.g. SWA tail-window), so blocks
                that can never serve a hit stay out of the prefix-cache hash
                map.
        """
        if num_cached_blocks >= num_full_blocks:
            # Full prefix hit (no new blocks to cache). Still apply directives so
            # a covering reuse refreshes the retention meta's expiry; otherwise a
            # purely-reused prefix never reaches the hook and its protection lapses.
            self._apply_retention_hook(request, blocks, num_full_blocks, block_size)
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert block_mask is None or len(block_mask) == len(new_full_blocks)
        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null or masked out when enabling sparse attention
            # like sliding window attention, or Mamba models with prefix-caching
            # in align mode. We skip null blocks here.
            if blk.is_null or (block_mask is not None and not block_mask[i]):
                continue
            # Eager: get_new_blocks resets the hash for every allocated block,
            # so it always arrives here cleared.
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]
            num_hash_tokens = (num_cached_blocks + i + 1) * block_size

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            if blk.block_hash is not None:
                # The only valid case where a "new full block" already has a
                # hash is partial->full promotion of the same cache block.
                assert (
                    blk.block_hash_num_tokens is not None
                    and blk.block_hash_num_tokens < num_hash_tokens
                )
                removed_hashes = self._remove_cached_block_hashes(blk)
                self._emit_block_removed_events(removed_hashes)
            self._insert_block_hash(
                block_hash_with_group_id,
                blk,
                num_tokens=num_hash_tokens,
            )
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        self._apply_retention_hook(request, blocks, num_full_blocks, block_size)

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null/masked-out blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                if block_mask is not None and not block_mask[i - num_cached_blocks]:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                self._build_block_stored_event(
                    request,
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    block_size=block_size,
                    kv_cache_group_id=kv_cache_group_id,
                    extra_keys_list=extra_keys_list,
                )
            )

    def _build_block_stored_event(
        self,
        request: Request,
        block_hashes: list[ExternalBlockHash] | None,
        parent_block_hash: ExternalBlockHash | None,
        start_token_idx: int,
        end_token_idx: int,
        block_size: int,
        kv_cache_group_id: int,
        extra_keys_list: list[tuple[Any, ...] | None],
    ) -> BlockStored:
        """Build a ``BlockStored`` KV event for ``request``.

        Shared by ``cache_full_blocks`` (newly cached blocks) and
        ``emit_cached_block_events`` (prefix-cache-reused blocks) so both emit
        identical event shapes for downstream consumers.
        """
        return BlockStored(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=request.all_token_ids[start_token_idx:end_token_idx],
            block_size=block_size,
            lora_id=request.lora_request.adapter_id if request.lora_request else None,
            medium=MEDIUM_GPU,
            lora_name=request.lora_request.name if request.lora_request else None,
            extra_keys=extra_keys_list if extra_keys_list else None,
            group_idx=kv_cache_group_id,
        )

    def emit_cached_block_events(
        self,
        request: Request,
        num_cached_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Generate BlockStored events for blocks reused from prefix cache.

        Unlike cache_full_blocks(), this does NOT modify block state —
        the blocks are already cached. It only generates events so that
        external consumers (e.g. gateway) can learn about reused blocks.

        Args:
            request: The request whose prefix cache blocks were reused.
            num_cached_blocks: Number of blocks that were cache hits.
            block_size: Number of tokens per block.
            kv_cache_group_id: The KV cache group ID.
        """
        if not self.enable_kv_cache_events or num_cached_blocks == 0:
            return

        block_hashes = resolve_block_hashes(
            request.block_hashes, self.hash_block_size, block_size
        )

        # Collect external hashes and extra_keys for cached blocks.
        cached_hashes: list[ExternalBlockHash] = []
        extra_keys_list: list[tuple[Any, ...] | None] = []
        curr_mm_idx = 0
        for i in range(num_cached_blocks):
            block_start = i * block_size
            block_end = block_start + block_size
            cached_hashes.append(maybe_convert_block_hash(block_hashes[i]))
            extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            extra_keys_list.append(extra_keys)

        if not cached_hashes:
            return

        # Prefix-cache hits always form a contiguous prefix starting at block 0,
        # so the first (and thus the whole group's) parent block hash is None.
        parent_block_hash: ExternalBlockHash | None = None
        start_token_idx = 0
        end_token_idx = num_cached_blocks * block_size

        logger.debug(
            "EmitCachedBlock event: block_size=%d, "
            "num_cached_blocks=%d, parent_block_hash=%s, "
            "token_ids_len=%d, group_idx=%s",
            block_size,
            num_cached_blocks,
            parent_block_hash,
            len(request.all_token_ids[start_token_idx:end_token_idx]),
            kv_cache_group_id,
        )

        self.kv_event_queue.append(
            self._build_block_stored_event(
                request,
                block_hashes=cached_hashes,
                parent_block_hash=parent_block_hash,
                start_token_idx=start_token_idx,
                end_token_idx=end_token_idx,
                block_size=block_size,
                kv_cache_group_id=kv_cache_group_id,
                extra_keys_list=extra_keys_list,
            )
        )

    def cache_partial_block(
        self,
        request: Request,
        block: KVCacheBlock,
        num_tokens: int,
        kv_cache_group_id: int,
        block_size: int,
    ) -> BlockHashWithGroupId | None:
        """Register a partial prefix-cache entry for an existing block.

        Prefix-cache keys normally identify full cache blocks. A partial entry
        makes an existing cache block reachable from a fine-grained prefix
        boundary inside that block without allocating or copying a new
        ``KVCacheBlock``.

        The partial entry is lookup metadata owned by ``block``. If ``block``
        has no primary hash, the key becomes its primary hash. If the block
        already has a primary hash, the partial entry is tracked in
        ``cached_block_hashes_by_block`` so eviction, reset, and promotion can
        remove every hash key that points to the block.

        Args:
            request: Request whose token IDs and block hashes define the
                partial entry.
            block: Existing cache block to make reachable from the partial
                prefix boundary.
            num_tokens: Prefix length represented by the partial entry. It
                must be a positive multiple of ``self.hash_block_size`` and
                cannot exceed the request's computed block hashes.
            kv_cache_group_id: KV cache group that owns the partial entry.
            block_size: Cache block size for the owning group. The partial
                entry hash itself is always the prefix-chain hash at
                ``num_tokens``; ``block_size`` is used to assert that the
                entry is partial within the owning cache block.

        Returns:
            The hash key with group ID if a partial entry can be registered;
            otherwise ``None`` for null blocks.
        """
        if block.is_null:
            return None

        assert block_size > self.hash_block_size
        assert block_size % self.hash_block_size == 0
        assert num_tokens % block_size != 0
        block_hash = self._get_partial_block_hash(request, num_tokens)
        num_hash_blocks = num_tokens // self.hash_block_size
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        already_cached = block.block_hash == block_hash_with_group_id or (
            self.cached_block_hash_to_block.contain(
                block_hash_with_group_id, block.block_id
            )
        )
        if (
            not already_cached
            and block.block_hash is not None
            and block.block_hash_num_tokens is not None
            and block.block_hash_num_tokens < num_hash_blocks * self.hash_block_size
        ):
            removed_hashes = self._remove_cached_block_hashes(block)
            self._emit_block_removed_events(removed_hashes)
        self._insert_block_hash(
            block_hash_with_group_id,
            block,
            num_tokens=num_hash_blocks * self.hash_block_size,
        )
        if self.enable_kv_cache_events and not already_cached:
            parent_hash, block_start = self._get_partial_block_parent_hash_and_start(
                request, num_tokens
            )
            parent_block_hash = (
                maybe_convert_block_hash(parent_hash)
                if parent_hash is not None
                else None
            )
            block_end = num_tokens
            curr_mm_idx = -1 if block_start > 0 else 0
            extra_keys, _ = generate_block_hash_extra_keys(
                request, block_start, block_end, curr_mm_idx
            )
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[maybe_convert_block_hash(block_hash)],
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[block_start:block_end],
                    block_size=block_end - block_start,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=[extra_keys],
                    group_idx=kv_cache_group_id,
                )
            )
        return block_hash_with_group_id

    def _get_partial_block_hash(
        self,
        request: Request,
        num_tokens: int,
    ) -> BlockHash:
        assert num_tokens % self.hash_block_size == 0
        num_hash_blocks = num_tokens // self.hash_block_size
        assert 0 < num_hash_blocks <= len(request.block_hashes)

        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        return request.block_hashes[num_hash_blocks - 1]

    def _get_partial_block_parent_hash_and_start(
        self,
        request: Request,
        num_tokens: int,
    ) -> tuple[BlockHash | None, int]:
        num_hash_blocks = num_tokens // self.hash_block_size
        parent_hash = (
            request.block_hashes[num_hash_blocks - 2] if num_hash_blocks > 1 else None
        )
        block_start = (num_hash_blocks - 1) * self.hash_block_size
        return parent_hash, block_start

    def _remove_cached_block_hashes(
        self,
        block: KVCacheBlock,
    ) -> list[BlockHashWithGroupId]:
        block_hashes: list[BlockHashWithGroupId] = []
        if block.block_hash is not None:
            block_hashes.append(block.block_hash)
        block_hashes.extend(self.cached_block_hashes_by_block.pop(block.block_id, ()))
        if not block_hashes:
            return []

        removed_hashes: list[BlockHashWithGroupId] = []
        for block_hash in block_hashes:
            if (
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ):
                removed_hashes.append(block_hash)
        block.reset_hash()
        return removed_hashes

    def _emit_block_removed_events(
        self,
        block_hashes: list[BlockHashWithGroupId],
    ) -> None:
        if not self.enable_kv_cache_events:
            return
        for block_hash in block_hashes:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )

    def _insert_block_hash(
        self,
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        num_tokens: int | None,
    ) -> None:
        if block.block_hash == block_hash_with_group_id:
            return

        if self.cached_block_hash_to_block.contain(
            block_hash_with_group_id, block.block_id
        ):
            return

        if block.block_hash is None:
            block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
        else:
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
                block_hash_with_group_id
            )
        self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)

    def move_block_hashes(
        self,
        src_block: KVCacheBlock,
        dst_block: KVCacheBlock,
    ) -> None:
        """Re-point ``src_block``'s prefix-cache entries to ``dst_block``.

        Used when the request owning ``src_block`` keeps writing into it
        : the prefix cache holds a private copy (``dst_block``)
        under the same hashes instead. Entries stay live; no events emitted.
        """
        assert dst_block.block_hash is None
        assert dst_block.block_id not in self.cached_block_hashes_by_block
        num_tokens = src_block.block_hash_num_tokens
        for block_hash in self._remove_cached_block_hashes(src_block):
            # `num_tokens` only applies to the first (primary) insertion.
            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)

    def _apply_retention_hook(
        self,
        request,
        blocks: list[KVCacheBlock],
        num_full_blocks: int,
        block_size: int,
    ) -> None:
        """Read retention directives from request.sampling_params.extra_args
        and apply them to the given full blocks. No-op when neither
        retention_directives nor retention_scope is present (zero overhead
        for non-retention requests)."""
        extra = getattr(request.sampling_params, "extra_args", None) or {}
        directives = extra.get("retention_directives")
        scope = extra.get("retention_scope")
        if directives is None and scope is None:
            return
        if directives:
            # A covers_output directive protects the generated tail. It carries
            # no explicit start; resolve it to [num_prompt_tokens, None) so it
            # pins the output range rather than defaulting to start=0 (which
            # would pin the whole prompt).
            num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
            if num_prompt_tokens is not None:
                directives = [
                    {**d, "start": num_prompt_tokens, "end": None}
                    if d.get("covers_output")
                    else d
                    for d in directives
                ]
        released = self.priority_eviction_queue.apply_directives(
            blocks[:num_full_blocks],
            directives or [],
            scope,
            block_size,
        )
        # A released block that is already free left the priority queue without
        # joining the LRU list, so nothing could ever allocate it again. Reuse
        # applies directives before touch() runs, which is exactly when a hit
        # block is free and still queued, so route those back the way
        # release_expired()'s blocks are routed. A referenced block needs no
        # routing: losing its sidecar sends it to the LRU list when it is freed.
        if released:
            freed = [
                self.blocks[block_id]
                for block_id in released
                if self.blocks[block_id].ref_cnt == 0
            ]
            if freed:
                self.free_block_queue.append_n(freed)

    def hold_preempted_blocks(self, blocks: Iterable[KVCacheBlock]) -> int:
        """Hold a preempted request's blocks so its resume can read them back.

        Freeing the request's blocks is how preemption makes room, and the ones
        it never claimed land in the LRU list -- where blocks freed afterwards
        queue up behind them and are reclaimed later, so the blocks a resume
        needs are spent before genuinely dead ones. Holding them reverses that
        order. Set the hold priority to 0 to turn this off.
        """
        priority = envs.VLLM_RETENTION_PREEMPT_HOLD_PRIORITY
        if priority <= 0:
            return 0
        limit = int(len(self.blocks) * envs.VLLM_RETENTION_PREEMPT_HOLD_FRAC)
        return self.priority_eviction_queue.hold_blocks(
            blocks,
            priority=priority,
            duration=envs.VLLM_RETENTION_PREEMPT_HOLD_S,
            limit=limit,
        )

    def refresh_retention_on_reuse(
        self,
        request: Request,
        hit_blocks: Sequence[list[KVCacheBlock]],
        block_sizes: Sequence[int],
    ) -> None:
        """Refresh retention protection for the blocks that served a cache hit.

        Applying directives only while caching leaves a reused block
        unprotected in two ways. Expiry releases protection without evicting,
        so a lapsed block keeps serving hits with no sidecar entry until some
        request happens to re-cache it. And when the same content sits in
        several cached blocks, the caching path refreshes the block it
        allocated, not the one that served the hit. Reuse is the strongest
        evidence a block is still needed, so refresh here, where the blocks
        that actually served the hit are known.

        Args:
            request: The reusing request, whose directives are applied.
            hit_blocks: Per-group hit blocks, each a prefix starting at token 0.
            block_sizes: Block size of each KV cache group, in the same order.
        """
        for group_blocks, block_size in zip(hit_blocks, block_sizes):
            if group_blocks:
                self._apply_retention_hook(
                    request, group_blocks, len(group_blocks), block_size
                )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        # Demote expired protections to the LRU tail rather than evicting them:
        # expiry means "protection released," not "evict now," so treating a
        # lapsed protection as freshly freed avoids mass-evicting a still-live
        # working set on a TTL underestimate.
        expired = [
            self.blocks[block_id]
            for block_id in self.priority_eviction_queue.release_expired()
        ]
        if expired:
            self.free_block_queue.append_n(expired)

        # LRU first, then top up from the priority queue (lowest priority
        # first); the while loop never runs when nothing is prioritized.
        num_from_free = min(num_blocks, self.free_block_queue.num_free_blocks)
        ret = self.free_block_queue.popleft_n(num_from_free)
        while len(ret) < num_blocks:
            evicted = self.priority_eviction_queue.pop_lowest()
            assert evicted is not None, "Priority queue empty but more blocks required"
            ret.append(evicted)

        # Eager: a priority-queue pop is a real eviction like an LRU pop, so
        # reset the hash at allocation.
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        evicted_hashes = self._remove_cached_block_hashes(block)
        if not evicted_hashes:
            # The block doesn't have hash, eviction is not needed
            return False

        self._emit_block_removed_events(evicted_hashes)
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for block in blocks:
            # ref_cnt=0 means the block is logically free; remove it from
            # whichever queue holds it (priority queue or LRU free list).
            if block.ref_cnt == 0 and not block.is_null:
                if block in self.priority_eviction_queue:
                    # A hold lasts until the block is read again, and this is
                    # that read, so it is spent. Suspending it instead would
                    # keep the entry for the rest of its window and go on
                    # protecting a block whose reader has moved on -- and would
                    # leave a hold standing in for a claim in what the sidecar
                    # reports. A real claim survives, as always.
                    if self.priority_eviction_queue.is_hold(block.block_id):
                        self.priority_eviction_queue.unprotect(block.block_id)
                    else:
                        self.priority_eviction_queue.suspend(block)
                elif (
                    block.prev_free_block is not None
                    or block.next_free_block is not None
                ):
                    self.free_block_queue.remove(block)
                # else: in neither queue (unexpected for a ref_cnt=0 block).
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Identify blocks with hash (LRU cache) and without it (will never match in APC)
        blocks_with_hash = []
        blocks_without_hash = []
        # Stamp monotonic time for the priority-queue heap tiebreak, with a
        # per-block nanosecond offset so pop_lowest evicts tail-first,
        # matching LRU order (ordered_blocks arrives tail->head). This avoids
        # an arbitrary block_id tiebreak that could strand a mid-prefix block
        # and break the cached chain; the offset is far smaller than real
        # inter-free gaps, so cross-request recency ordering is unaffected.
        now = time.monotonic()
        for pos, block in enumerate(ordered_blocks):
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                # Protected blocks go to the priority queue (try_insert falls
                # through to LRU for the rest). Beyond the budget, drop the
                # protection instead so a future reallocation can't inherit
                # stale metadata.
                pq = self.priority_eviction_queue
                if pq.num_blocks < self.retention_budget_blocks:
                    if pq.try_insert(block, last_freed_time=now + pos * 1e-9):
                        continue
                else:
                    pq.unprotect(block.block_id)
                if block.block_hash is None:
                    blocks_without_hash.append(block)
                else:
                    blocks_with_hash.append(block)

        # Blocks without hash always get evicted first - prepend them last to the tail
        self.free_block_queue.prepend_n(blocks_without_hash)
        self.free_block_queue.append_n(blocks_with_hash)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)
            self.priority_eviction_queue.unprotect(block_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block.clear()

        # Return priority-queue blocks to the LRU free list before clearing:
        # clear() alone leaks them, since they belong to no other queue.
        while (block := self.priority_eviction_queue.pop_lowest()) is not None:
            self.free_block_queue.append(block)
        self.priority_eviction_queue.clear()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks (LRU + prioritized).
        """
        return (
            self.free_block_queue.num_free_blocks
            + self.priority_eviction_queue.num_blocks
        )

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority-based eviction queue with sidecar storage of per-block
retention metadata."""

import heapq
import time
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(slots=True)
class RetentionMeta:
    priority: int
    expiry: float | None
    scope: str | None
    last_freed_time: float
    # A hold is not a client's retention claim: the scheduler installs it on the
    # blocks of a request it preempts, because that request re-reads them when it
    # resumes. It stands in only until something real arrives, so any covering
    # directive replaces it regardless of priority (see apply_directives), and it
    # is short-lived by construction.
    hold: bool = False


def _later_expiry(a: float | None, b: float | None) -> float | None:
    """The later of two expiries, where None means "never expires" and wins."""
    if a is None or b is None:
        return None
    return max(a, b)


class PriorityEvictionQueue:
    def __init__(self) -> None:
        self._meta: dict[int, RetentionMeta] = {}
        self._heap: list[tuple[int, float, int, int, KVCacheBlock]] = []
        self._in_queue: set[int] = set()
        # Per-block generation counter. try_insert bumps it and stamps the
        # pushed tuple; pop_lowest skips tuples with a stale generation. Guards
        # against a suspend()+re-insert leaving an outdated tuple that would
        # evict by the old (priority, last_freed_time) and invert the order.
        self._gen: dict[int, int] = {}

    @property
    def num_blocks(self) -> int:
        return len(self._in_queue)

    def __contains__(self, block: KVCacheBlock) -> bool:
        return block.block_id in self._in_queue

    def try_insert(
        self,
        block: KVCacheBlock,
        last_freed_time: float | None = None,
    ) -> bool:
        """Insert the block's sidecar entry into the heap and return True.
        Return False (caller routes to LRU) when there is no entry or it has
        expired; the expired case drops the sidecar. last_freed_time, if given,
        overrides the stored value so the heap tiebreak reflects the most-recent
        free."""
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False
        if meta.expiry is not None and meta.expiry <= time.monotonic():
            # Expired-on-free: drop sidecar, route to LRU free list.
            self._meta.pop(block.block_id, None)
            return False
        if last_freed_time is not None:
            meta.last_freed_time = last_freed_time
        gen = self._gen.get(block.block_id, 0) + 1
        self._gen[block.block_id] = gen
        heapq.heappush(
            self._heap,
            (meta.priority, meta.last_freed_time, gen, block.block_id, block),
        )
        self._in_queue.add(block.block_id)
        return True

    def suspend(self, block: KVCacheBlock) -> None:
        """Drop the block from the eviction-candidate set (_in_queue) only;
        the sidecar (_meta) is KEPT so protection is restored on the next
        free. The stale heap tuple is skipped lazily at pop_lowest. Contrast
        unprotect(), which removes the protection record."""
        self._in_queue.discard(block.block_id)

    def pop_lowest(self) -> KVCacheBlock | None:
        """Pop and return the lowest-priority block.

        Skips stale tuples: those whose block_id has left _in_queue (suspend /
        release_expired) and those whose generation is outdated (a newer
        try_insert superseded them). Expired entries are not handled here —
        callers must invoke release_expired() first. Returns None when no
        live entries remain."""
        while self._heap:
            _, _, gen, block_id, block = heapq.heappop(self._heap)
            if block_id not in self._in_queue:
                continue
            if gen != self._gen.get(block_id):
                # Superseded by a newer insert for the same block_id.
                continue
            self._in_queue.discard(block_id)
            self._meta.pop(block_id, None)
            return block
        return None

    def release_expired(self) -> list[int]:
        """Release protection from all expired entries and return their
        block_ids for the caller to route to the LRU free list (expiry =
        "protection released", not "evict now"). Stale heap tuples are
        cleaned up at the next pop_lowest."""
        now = time.monotonic()
        drained: list[int] = []
        for block_id in list(self._in_queue):
            meta = self._meta.get(block_id)
            if meta is not None and meta.expiry is not None and meta.expiry <= now:
                self._in_queue.discard(block_id)
                self._meta.pop(block_id, None)
                drained.append(block_id)
        return drained

    def unprotect(self, block_id: int) -> None:
        """Permanently drop the block's protection: pop its sidecar (_meta)
        and discard it from _in_queue. Called when the block's hash is reset
        or it is evicted from the prefix cache. Unlike suspend(), the
        protection record does NOT survive."""
        self._meta.pop(block_id, None)
        self._in_queue.discard(block_id)

    def hold_blocks(
        self,
        blocks: Iterable[KVCacheBlock],
        priority: int,
        duration: float,
        limit: int,
    ) -> int:
        """Hold blocks a preempted request will re-read, and return how many.

        A preempted request restarts its prefill and reads back the blocks it
        already filled, so those blocks are about to be needed again -- but they
        carry no claim of their own, which puts them in the LRU list where
        anything freed later is reclaimed after them. A hold moves them into the
        queue so genuinely dead blocks are spent first.

        Only blocks with no entry are held, so a client's claim is never
        overwritten, and only up to ``limit`` total blocks are ever held, so a
        few large preemptions cannot pin the cache.

        Args:
            blocks: The request's blocks, prefix first — a resumed request can
                only use a contiguous prefix, so that is what is held.
            priority: Priority for the hold.
            duration: Seconds to hold for. Short: it only has to outlast the
                wait for this request to be scheduled again.
            limit: Ceiling on the total number of blocks held at once.
        """
        expiry = time.monotonic() + duration
        held = 0
        for block in blocks:
            if len(self._meta) >= limit:
                break
            if block.is_null or block.block_id in self._meta:
                continue
            self._meta[block.block_id] = RetentionMeta(
                priority=priority,
                expiry=expiry,
                scope=None,
                last_freed_time=0.0,
                hold=True,
            )
            held += 1
        return held

    def clear(self) -> None:
        """Drop all sidecar entries and heap state."""
        self._meta.clear()
        self._heap.clear()
        self._in_queue.clear()
        self._gen.clear()

    def apply_directives(
        self,
        blocks: list[KVCacheBlock],
        directives: list[dict],
        scope: str | None,
        block_size: int,
    ) -> None:
        """For each full block, find the highest-priority overlapping
        directive and update the sidecar entry under these rules:

        - Priority 1-100 protects; priority 0 releases.
        - Escalation (new > current priority): any caller may raise priority
          and takes ownership of the block. The expiry only moves later, so
          raising a block's priority never shortens a hold already in place.
          Only its owner shortens a block's hold, by naming it again.
        - Downgrade or refresh (new <= current priority): only the current
          owner may do this.
        - Release (priority 0): only the current owner may drop the entry, the
          same restriction as a downgrade. A caller that no longer needs a
          block says so; a block nobody renews also expires on its own.
        - No matching directive: the block is left alone. Saying nothing about
          a block is not a request to unprotect it, and directives without an
          explicit priority are ignored rather than read as a release -- when
          silence meant release, one turn dropped the protection another turn
          had just placed on a block it was actively reusing.
        """
        now = time.monotonic()
        for idx, block in enumerate(blocks):
            if block.is_null:
                continue
            token_start = idx * block_size
            token_end = token_start + block_size
            best_priority = -1
            best_duration: float | None = None
            for d in directives:
                d_start = d.get("start", 0)
                d_end = d.get("end")
                if d_end is not None and d_end <= token_start:
                    continue
                if d_start >= token_end:
                    continue
                p = d.get("priority")
                if p is None:
                    continue
                if p > best_priority:
                    best_priority = p
                    best_duration = d.get("duration")

            current = self._meta.get(block.block_id)
            if current is not None and current.hold:
                # A scheduler hold yields to any real claim: treat it as absent so
                # the covering directive lands instead of being read as a
                # downgrade and dropped, which would leave the block on the
                # hold's short expiry rather than the one its owner asked for.
                current = None

            if best_priority < 0:
                # No matching directive: leave the block's protection as it is.
                continue

            if best_priority == 0:
                # Explicit release. Restricted to the owner, like a downgrade:
                # otherwise one scope could drop protection another scope is
                # relying on. Nothing to do when the block is unprotected.
                if scope is not None and current is not None and current.scope == scope:
                    self.unprotect(block.block_id)
                continue

            expiry = now + best_duration if best_duration is not None else None
            current_priority = current.priority if current is not None else -1
            if best_priority > current_priority:
                # Escalation: any caller may raise priority and takes ownership.
                # The hold only grows — raising a block's priority must not cut
                # short a longer one someone else is already relying on.
                self._meta[block.block_id] = RetentionMeta(
                    priority=best_priority,
                    expiry=(
                        expiry
                        if current is None
                        else _later_expiry(expiry, current.expiry)
                    ),
                    scope=scope,
                    last_freed_time=current.last_freed_time if current else 0.0,
                )
            elif current is not None and best_priority == current_priority:
                # Equal-priority covering reuse (any scope): refresh the expiry so a
                # shared prefix reused across sessions does not lapse. Keep the
                # original owner and priority — a cross-scope caller may only EXTEND
                # the hold, never downgrade or steal ownership.
                self._meta[block.block_id] = RetentionMeta(
                    priority=current.priority,
                    expiry=_later_expiry(expiry, current.expiry),
                    scope=current.scope,
                    last_freed_time=current.last_freed_time,
                )
            elif current is not None and scope is not None and current.scope == scope:
                # Same scope: owner may downgrade or refresh.
                self._meta[block.block_id] = RetentionMeta(
                    priority=best_priority,
                    expiry=expiry,
                    scope=scope,
                    last_freed_time=current.last_freed_time,
                )
            # Non-owner downgrade: silently ignored.

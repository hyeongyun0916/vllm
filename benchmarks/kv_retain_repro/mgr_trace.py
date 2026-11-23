"""Drive a small-pool KVCacheManager with a seeded hint-free multi-session workload and print one line per step:
cache hits, allocated block ids, and the LRU free-list order. Run it once against each vLLM tree (PYTHONPATH picks the
tree) and diff the outputs: identical output means identical eviction behaviour.
usage: cd <vllm tree> && PYTHONPATH=<vllm tree> python mgr_trace.py <seed> <steps> [num_blocks]"""
import random
import sys

from tests.v1.core.test_prefix_caching import (
    make_kv_cache_config,
    make_kv_cache_manager,
    make_request as _make_request,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash

BS = 16
seed, steps = int(sys.argv[1]), int(sys.argv[2])
num_blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 40
init_none_hash(sha256)
manager = make_kv_cache_manager(
    make_kv_cache_config(BS, num_blocks), max_model_len=65536, enable_caching=True, hash_block_size=BS
)
pool = manager.block_pool
evictions = 0
_evict = type(pool)._maybe_evict_cached_block


def _counting_evict(self, block, *args, **kwargs):
    global evictions
    evicted = _evict(self, block, *args, **kwargs)
    evictions += bool(evicted)
    return evicted


type(pool)._maybe_evict_cached_block = _counting_evict


def request(rid, tokens):
    return _make_request(rid, tokens, BS, sha256)   # SamplingParams(max_tokens=17), eos 100, as before


def free_order():
    q, out = pool.free_block_queue, []
    b = q.fake_free_list_head.next_free_block
    while b is not q.fake_free_list_tail:
        out.append(b.block_id)
        b = b.next_free_block
    return out


rnd = random.Random(seed)
system = [90000 + i for i in range(2 * BS)]
sessions = [list(system) for _ in range(6)]
in_flight = []
for step in range(steps):
    if in_flight and (len(in_flight) >= 4 or rnd.random() < 0.45):
        r = in_flight.pop(rnd.randrange(len(in_flight)))
        manager.free(r)
        print(f"{step} free {r.request_id} lru={free_order()}")
        continue
    sid = rnd.randrange(len(sessions))
    if rnd.random() < 0.1:
        sessions[sid] = list(system)                       # new conversation in this slot
    sessions[sid] += [sid * 100000 + step] * rnd.randint(1, 4 * BS)
    r = request(f"r{step}", list(sessions[sid]))
    computed, num_computed, _ = manager.get_computed_blocks(r)
    out = manager.allocate_slots(r, r.num_tokens - num_computed, num_computed, computed)
    if out is None:
        print(f"{step} reject hits={num_computed}")
        continue
    ids = manager.get_block_ids(r.request_id)[0]
    in_flight.append(r)
    print(f"{step} alloc hits={num_computed} blocks={ids} lru={free_order()}")
print(f"end evictions={evictions}")

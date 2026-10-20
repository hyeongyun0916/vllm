"""Drive a small-pool KVCacheManager with a seeded hint-free multi-session workload and print one line per step:
cache hits, allocated block ids, and the LRU free-list order. Run it once against each vLLM tree (PYTHONPATH picks the
tree) and diff the outputs: identical output means identical eviction behaviour.
usage: PYTHONPATH=<vllm tree> python mgr_trace.py <seed> <steps> [num_blocks]"""
import random
import sys

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

BS = 16
seed, steps = int(sys.argv[1]), int(sys.argv[2])
num_blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 40
init_none_hash(sha256)
config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
    KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32))])
manager = KVCacheManager(config, max_model_len=65536, enable_caching=True, hash_block_size=BS, scheduler_block_size=BS)
pool = manager.block_pool
hasher = get_request_block_hasher(BS, sha256)
evictions = 0
_evict = type(pool)._maybe_evict_cached_block


def _counting_evict(self, block, *args, **kwargs):
    global evictions
    evicted = _evict(self, block, *args, **kwargs)
    evictions += bool(evicted)
    return evicted


type(pool)._maybe_evict_cached_block = _counting_evict


def request(rid, tokens):
    sp = SamplingParams(max_tokens=17)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(request_id=rid, prompt_token_ids=tokens, sampling_params=sp, pooling_params=None, block_hasher=hasher)


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

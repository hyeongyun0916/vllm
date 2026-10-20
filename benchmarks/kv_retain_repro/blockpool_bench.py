"""CPU microbenchmark of the no-hint (LRU) KV-cache manager path: multi-turn sessions with shared prefixes over a pool
smaller than the working set, so lookups hit, blocks get cached, touched, freed and evicted -- no kv.retain hints anywhere.
Run from a vLLM source tree root (imports tests.v1.core.test_prefix_caching helpers of that tree):
  cd <tree> && PYTHONPATH=<tree> python blockpool_bench.py [reps]
Reports median per-request time per manager op and the total, so two trees can be compared."""
import statistics, sys, time
from tests.v1.core.test_prefix_caching import make_kv_cache_config, make_kv_cache_manager, make_request
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.utils.hashing import sha256  # BENCH_V3: block hashes are bytes

BLOCK = 16; POOL_BLOCKS = 1500; SESSIONS = 32; TURNS = 12; SHARED_BLOCKS = 12; GROW_BLOCKS = 8
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def run_once():
    manager = make_kv_cache_manager(make_kv_cache_config(BLOCK, POOL_BLOCKS), max_model_len=131072, enable_caching=True, hash_block_size=BLOCK)
    hash_fn = sha256
    shared = list(range(SHARED_BLOCKS * BLOCK))
    t_lookup = t_alloc = t_free = 0.0; n = 0; hits = 0
    for turn in range(TURNS):
        for s in range(SESSIONS):
            prompt = shared + [10_000 + s * 100_000 + i for i in range((turn + 1) * GROW_BLOCKS * BLOCK)]
            req = make_request(f"{s}-{turn}", prompt, BLOCK, hash_fn)
            t0 = time.perf_counter()
            computed, num_computed, _ = manager.get_computed_blocks(req)
            t1 = time.perf_counter()
            blocks = manager.allocate_slots(req, len(prompt) - num_computed, num_computed, computed)
            assert blocks is not None, "pool too small for one request"
            t2 = time.perf_counter()
            manager.free(req)
            t3 = time.perf_counter()
            t_lookup += t1 - t0; t_alloc += t2 - t1; t_free += t3 - t2; n += 1; hits += num_computed
    return dict(lookup=t_lookup / n * 1e6, alloc=t_alloc / n * 1e6, free=t_free / n * 1e6, total=(t_lookup + t_alloc + t_free) / n * 1e6, hit_tokens=hits)

init_none_hash(sha256)
runs = [run_once() for _ in range(REPS)]
med = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
import vllm; print(f"vllm {vllm.__version__} from {vllm.__file__}")
print(f"requests/rep {SESSIONS * TURNS}, pool {POOL_BLOCKS} blocks, hit tokens/rep {med['hit_tokens']:.0f}")
print(f"median us/request: lookup {med['lookup']:.1f}  alloc {med['alloc']:.1f}  free {med['free']:.1f}  total {med['total']:.1f}  (reps {REPS}: totals {' '.join(f'{r['total']:.1f}' for r in runs)})")

# Reproduction scripts for vllm-project/vllm#38514 (hint-free path)

Evidence that a request without `kv_hints` sees the same KV-cache behaviour and cost as the PR's base commit.
Both scripts need two vLLM source trees: the PR head and its base (`c0b6f74bee`), e.g.

    git worktree add ../vllm-base c0b6f74bee
    git worktree add ../vllm-pr   <PR head>

CPU only; no GPU or model weights. Use one Python env with vLLM's deps installed; `PYTHONPATH` selects the tree.

## 1. Differential trace of the KV-cache manager

`mgr_trace.py` drives a 40-block `KVCacheManager` with a seeded hint-free workload (six sessions that share a system
prefix and grow turn by turn, up to four requests in flight, occasional session restarts) through
`get_computed_blocks` -> `allocate_slots` -> `free`, and prints one line per step: cache hits, allocated block ids and
the LRU free-list order, plus the eviction count at the end. `run_mgr_diff.sh` runs it on both trees for many seeds
and compares the traces line by line.

    PY=python bash benchmarks/kv_retain_repro/run_mgr_diff.sh ../vllm-pr ../vllm-base 100 1500

## 2. CPU cost of the manager path

`blockpool_bench.py`: 32 sessions x 12 turns over a 1,500-block pool, no hints; median microseconds per request for
lookup + allocate + free. Run from each tree's root (it imports that tree's test helpers):

    cd ../vllm-base && PYTHONPATH=$PWD taskset -c 2 python ../vllm-pr/benchmarks/kv_retain_repro/blockpool_bench.py 7
    cd ../vllm-pr   && PYTHONPATH=$PWD taskset -c 2 python benchmarks/kv_retain_repro/blockpool_bench.py 7

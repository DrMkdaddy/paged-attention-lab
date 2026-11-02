"""Quantify what paging buys: admitted sequences under a fixed budget, and the
read overhead paid for it.

Run: python3 bench/bench.py [--out results/paged_attention.json]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paged_kv import PagedKVCache, attention, paged_attention  # noqa: E402


def admitted(lengths, block_size, budget_blocks, paged):
    used = 0
    n = 0
    longest = math.ceil(max(lengths) / block_size)
    for length in lengths:
        need = math.ceil(length / block_size) if paged else longest
        if used + need > budget_blocks:
            break
        used += need
        n += 1
    return n, used


def time_call(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/paged_attention.json")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(13)
    results = {"block_size": args.block_size, "dim": args.dim}

    # 1. admitted sequences under a fixed budget, same traffic
    lengths = list(rng.integers(50, 2000, size=64))
    budget = 4096
    paged_n, paged_used = admitted(lengths, args.block_size, budget, paged=True)
    contig_n, contig_used = admitted(lengths, args.block_size, budget, paged=False)
    total_tokens = sum(lengths[:paged_n])
    results["admission"] = {
        "budget_blocks": budget,
        "block_size": args.block_size,
        "lengths": [int(x) for x in lengths[:8]],
        "paged_sequences_admitted": paged_n,
        "contiguous_sequences_admitted": contig_n,
        "paged_blocks_used": paged_used,
        "contiguous_blocks_used": contig_used,
        "paged_tokens_served": int(total_tokens),
        "ratio": round(paged_n / contig_n, 2) if contig_n else None,
    }

    # 2. reserved-vs-used waste on a single mixed batch
    waste_rows = []
    for n_seq in (4, 16, 64):
        lens = rng.integers(50, 2000, size=n_seq)
        paged_slots = sum(math.ceil(int(x) / args.block_size) * args.block_size for x in lens)
        contig_slots = n_seq * int(lens.max())
        waste_rows.append({
            "sequences": n_seq,
            "paged_slots": int(paged_slots),
            "contiguous_slots": int(contig_slots),
            "saving_pct": round(100 * (1 - paged_slots / contig_slots), 1),
        })
    results["waste"] = waste_rows

    # 3. read overhead: paged gather+attention vs one contiguous call
    overhead_rows = []
    for n in (64, 256, 1024):
        k = rng.normal(size=(n, args.dim))
        v = rng.normal(size=(n, args.dim))
        q = rng.normal(size=(1, args.dim))

        cache = PagedKVCache(num_blocks=math.ceil(n / args.block_size) + 1,
                             block_size=args.block_size, dim=args.dim)
        cache.add_sequence(0)
        cache.append(0, k, v)

        t_contig = time_call(lambda: attention(q, k, v))
        t_paged = time_call(lambda: paged_attention(cache, 0, q))
        overhead_rows.append({
            "tokens": n,
            "contiguous_ms": round(t_contig * 1000, 3),
            "paged_ms": round(t_paged * 1000, 3),
            "overhead_x": round(t_paged / t_contig, 2) if t_contig else None,
        })
    results["read_overhead"] = overhead_rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

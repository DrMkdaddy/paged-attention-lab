# paged-attention-lab

KV cache stored in fixed-size blocks with a per-sequence block table, and
attention that reads through that table instead of a contiguous slab.

**Why:** a contiguous cache reserves `max_len` per sequence. Paging allocates a
block at a time and lets sequences share prefix blocks, so the same memory holds
more concurrent sequences. The trade is that attention gathers K/V through an
indirection instead of a straight pointer.

## Result

Verified bit-exact against contiguous attention, then measured.

| Metric | Value |
|---|---|
| Paged vs contiguous output, 13-token sequence, token-at-a-time append | max abs error **0.00e+00** |
| Sequences admitted under a fixed 4096-block budget, same arrival order | **54 vs 32** (1.69x) |
| Tokens served in that window | 63,782 |
| Slot waste avoided, 4-64 sequences, lengths 50-2000 | **23.6% to 58.2%** |
| Read overhead vs one fused contiguous call | 5.5x at 64 tokens, 25.8x at 1024 |

The read overhead is real and expected: `paged_attention` is a Python loop over
blocks, while `attention` is a single NumPy matmul. It is the cost the design
trades for capacity, and it is what a fused kernel exists to remove.

## What is implemented

- `BlockAllocator` — fixed pool, explicit out-of-blocks error, double-free detection.
- `PagedKVCache` — block table per sequence, append across block boundaries,
  reference counting, fork with copy-on-write so two sequences share a prefix.
- `paged_attention` — attention that indexes `(block_table[pos // block_size],
  pos % block_size)` and never materialises a slab.
- `attention` — the contiguous baseline used for parity.

## Tests

```
$ python3 tests/test_paged_attention.py
ok   paged attention matches contiguous (max abs err 0.00e+00)
ok   multi-token append across block boundaries matches
ok   fork shares blocks without allocating
ok   copy-on-write allocates a private block
ok   parent KV is untouched after the fork writes
ok   paging reserves 2816 slots vs 12288 contiguous (77.1% less)
...
all passed (0 failure(s))
```

## Reproduce

```bash
python3 tests/test_paged_attention.py
python3 bench/bench.py            # writes results/paged_attention.json
```

## Scope

One head, one layer, float64, CPU. This is the cache and the gather, not a fused
GPU kernel, and not a serving engine. The numbers above are about capacity and
correctness under a fixed block budget, not tokens per second.

## License

MIT

"""Tests: paged attention must equal contiguous attention, and paging must save
real memory. Any mismatch here invalidates every number the benchmark produces.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paged_kv import (  # noqa: E402
    BlockAllocator, OutOfBlocks, PagedKVCache, attention, paged_attention,
)

FAILS = 0


def ok(cond, label):
    global FAILS
    if cond:
        print(f"ok   {label}")
    else:
        FAILS += 1
        print(f"FAIL {label}")


def test_allocator():
    a = BlockAllocator(4, 8)
    blocks = a.allocate(3)
    ok(len(blocks) == 3 and a.free_blocks == 1, "allocator hands out the requested blocks")
    a.free(blocks)
    ok(a.free_blocks == 4, "freeing returns every block")
    try:
        a.allocate(5)
        ok(False, "over-allocation must raise")
    except OutOfBlocks:
        ok(True, "over-allocation raises OutOfBlocks")
    try:
        a.free([blocks[0]])
        a.free([blocks[0]])
        ok(False, "double free must raise")
    except ValueError:
        ok(True, "double free raises")


def test_parity_contiguous():
    """Same tokens, same queries, two storage layouts, identical outputs."""
    rng = np.random.default_rng(7)
    dim, block_size, n = 16, 4, 13
    k = rng.normal(size=(n, dim))
    v = rng.normal(size=(n, dim))
    q = rng.normal(size=(n, dim))

    cache = PagedKVCache(num_blocks=16, block_size=block_size, dim=dim)
    cache.add_sequence(1)
    for i in range(n):                     # one token at a time, worst case
        cache.append(1, k[i:i + 1], v[i:i + 1])

    got = paged_attention(cache, 1, q)
    want = attention(q, k, v, causal=True)
    err = float(np.max(np.abs(got - want)))
    ok(err < 1e-12, f"paged attention matches contiguous (max abs err {err:.2e})")


def test_parity_batched_append():
    rng = np.random.default_rng(11)
    dim, block_size, n = 8, 5, 17
    k = rng.normal(size=(n, dim))
    v = rng.normal(size=(n, dim))
    q = rng.normal(size=(4, dim))

    cache = PagedKVCache(num_blocks=32, block_size=block_size, dim=dim)
    cache.add_sequence(2)
    cache.append(2, k[:9], v[:9])          # partial block boundary
    cache.append(2, k[9:], v[9:])          # crosses another boundary

    got = paged_attention(cache, 2, q)
    want = attention(q, k, v, causal=True)
    ok(float(np.max(np.abs(got - want))) < 1e-12, "multi-token append across block boundaries matches")


def test_fork_and_copy_on_write():
    rng = np.random.default_rng(3)
    dim, block_size = 8, 4
    prefix = rng.normal(size=(6, dim))

    cache = PagedKVCache(num_blocks=16, block_size=block_size, dim=dim)
    cache.add_sequence("a")
    cache.append("a", prefix, prefix)
    free_before = cache.alloc.free_blocks

    cache.fork("a", "b")
    ok(cache.alloc.free_blocks == free_before, "fork shares blocks without allocating")

    new_k = rng.normal(size=(1, dim))
    cache.append("b", new_k, new_k)
    ok(cache.lengths["a"] == 6, "writing to the fork does not change the parent length")
    ok(cache.alloc.free_blocks < free_before, "copy-on-write allocates a private block")

    ka, _ = cache.gather("a")
    ok(np.allclose(ka, prefix), "parent KV is untouched after the fork writes")


def test_memory_saving():
    """Contiguous reserves max_len per sequence; paging reserves only whole blocks."""
    block_size, dim, max_len = 16, 8, 2048
    lengths = [17, 300, 900, 41, 1500, 5]
    total_blocks = sum(math.ceil(n / block_size) for n in lengths)
    cache = PagedKVCache(num_blocks=total_blocks, block_size=block_size, dim=dim)
    rng = np.random.default_rng(5)
    for i, n in enumerate(lengths):
        cache.add_sequence(i)
        k = rng.normal(size=(n, dim))
        cache.append(i, k, k)

    paged_slots = cache.alloc.used_blocks * block_size
    contiguous_slots = cache.reserved_slots_contiguous(max_len)
    ok(cache.alloc.free_blocks == 0, "paged cache fits the exact block count")
    ok(paged_slots < contiguous_slots,
       f"paging reserves {paged_slots} slots vs {contiguous_slots} contiguous "
       f"({100 * (1 - paged_slots / contiguous_slots):.1f}% less)")


def test_drop_frees_everything():
    dim, block_size = 4, 4
    cache = PagedKVCache(num_blocks=10, block_size=block_size, dim=dim)
    cache.add_sequence(1)
    cache.append(1, np.ones((9, dim)), np.ones((9, dim)))
    cache.drop_sequence(1)
    ok(cache.alloc.free_blocks == 10, "dropping a sequence returns all of its blocks")


if __name__ == "__main__":
    test_allocator()
    test_parity_contiguous()
    test_parity_batched_append()
    test_fork_and_copy_on_write()
    test_memory_saving()
    test_drop_frees_everything()
    print(f"\n{'FAILED' if FAILS else 'all passed'} ({FAILS} failure(s))")
    raise SystemExit(1 if FAILS else 0)

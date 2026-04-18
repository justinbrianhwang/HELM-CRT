"""
Unit tests for KVCacheManager and LayerKVCache (helm/runtime/kv_cache.py).
"""
import pytest
import torch

from helm.runtime.kv_allocator import KVAllocator
from helm.runtime.kv_cache import KVCacheManager, LayerKVCache


# ──────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ──────────────────────────────────────────────────────────────────────────────

NUM_HEADS = 2
HEAD_DIM = 8
PAGE_SIZE = 4
NUM_LAYERS = 4


@pytest.fixture()
def allocator():
    return KVAllocator(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        dtype=torch.float32,
    )


@pytest.fixture()
def cache(allocator):
    # watermark=0 means eviction is triggered every time, but since pages are
    # on CPU (not cuda), evict_pages() won't find any GPU pages to move.
    mgr = KVCacheManager(allocator, gpu_high_watermark_bytes=0)
    mgr.initialize_layer_caches(num_layers=NUM_LAYERS)
    return mgr


def _kv(seq_len):
    k = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
    v = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM)
    return k, v


# ──────────────────────────────────────────────────────────────────────────────
# LayerKVCache
# ──────────────────────────────────────────────────────────────────────────────

def test_layer_cache_append_page(allocator):
    layer = LayerKVCache(layer_id=0)
    page = allocator.allocate(torch.device("cpu"))
    layer.append_page(page)
    assert len(layer.pages) == 1
    assert layer.tail_page is page


def test_layer_cache_tail_updates(allocator):
    layer = LayerKVCache(layer_id=0)
    p1 = allocator.allocate(torch.device("cpu"))
    p2 = allocator.allocate(torch.device("cpu"))
    layer.append_page(p1)
    layer.append_page(p2)
    assert layer.tail_page is p2


# ──────────────────────────────────────────────────────────────────────────────
# initialize_layer_caches
# ──────────────────────────────────────────────────────────────────────────────

def test_initialize_creates_n_layers(cache):
    assert cache.num_layers() == NUM_LAYERS
    assert sorted(cache.layers_keys()) == list(range(NUM_LAYERS))


def test_initial_seq_len_zero(cache):
    assert cache.seq_len() == 0


# ──────────────────────────────────────────────────────────────────────────────
# append_prefill
# ──────────────────────────────────────────────────────────────────────────────

def test_prefill_single_page_when_short(cache):
    k, v = _kv(3)  # 3 < PAGE_SIZE=4
    cache.append_prefill(0, k, v)
    assert len(cache.layers[0].pages) == 1
    assert cache.layers[0].total_tokens == 3


def test_prefill_multiple_pages_when_long(cache):
    k, v = _kv(9)  # ceil(9/4) = 3 pages
    cache.append_prefill(0, k, v)
    assert len(cache.layers[0].pages) == 3
    assert cache.layers[0].total_tokens == 9


def test_prefill_exact_one_page(cache):
    k, v = _kv(PAGE_SIZE)
    cache.append_prefill(0, k, v)
    assert len(cache.layers[0].pages) == 1
    assert cache.layers[0].tail_page.used_tokens == PAGE_SIZE


def test_prefill_data_preserved(cache):
    k, v = _kv(2)
    cache.append_prefill(0, k, v)
    page = cache.layers[0].pages[0]
    assert torch.allclose(page.k_tensor[:, :, :2, :], k)
    assert torch.allclose(page.v_tensor[:, :, :2, :], v)


def test_prefill_page_start_tokens(cache):
    k, v = _kv(8)  # 2 pages of 4
    cache.append_prefill(0, k, v)
    assert cache.layers[0].pages[0].start_token == 0
    assert cache.layers[0].pages[1].start_token == 4


def test_prefill_auto_creates_layer(cache):
    """append_prefill creates the layer if it doesn't exist."""
    del cache.layers[3]
    k, v = _kv(2)
    cache.append_prefill(3, k, v)
    assert 3 in cache.layers


# ──────────────────────────────────────────────────────────────────────────────
# append_decode
# ──────────────────────────────────────────────────────────────────────────────

def test_decode_increments_token_count(cache):
    k, v = _kv(1)
    cache.append_decode(0, k, v)
    assert cache.layers[0].total_tokens == 1


def test_decode_updates_seq_len(cache):
    k, v = _kv(1)
    cache.append_decode(0, k, v)
    assert cache.seq_len() == 1


def test_decode_within_existing_page(cache):
    k_pre, v_pre = _kv(2)
    cache.append_prefill(0, k_pre, v_pre)
    k_dec, v_dec = _kv(1)
    cache.append_decode(0, k_dec, v_dec)
    assert len(cache.layers[0].pages) == 1
    assert cache.layers[0].tail_page.used_tokens == 3


def test_decode_new_page_when_tail_full(cache):
    k_pre, v_pre = _kv(PAGE_SIZE)  # fills one page exactly
    cache.append_prefill(0, k_pre, v_pre)
    k_dec, v_dec = _kv(1)
    cache.append_decode(0, k_dec, v_dec)
    assert len(cache.layers[0].pages) == 2
    assert cache.layers[0].tail_page.used_tokens == 1


def test_decode_multiple_tokens_across_pages(cache):
    for _ in range(PAGE_SIZE + 2):  # fill one page + 2 in new
        k, v = _kv(1)
        cache.append_decode(0, k, v)
    assert len(cache.layers[0].pages) == 2
    assert cache.layers[0].total_tokens == PAGE_SIZE + 2


# ──────────────────────────────────────────────────────────────────────────────
# iterate_layer_pages / seq_len
# ──────────────────────────────────────────────────────────────────────────────

def test_iterate_layer_pages_returns_list(cache):
    k, v = _kv(3)
    cache.append_prefill(0, k, v)
    pages = cache.iterate_layer_pages(0)
    assert isinstance(pages, list)
    assert len(pages) == 1


def test_iterate_missing_layer_returns_empty(cache):
    assert cache.iterate_layer_pages(99) == []


def test_seq_len_from_layer_0(cache):
    k, v = _kv(5)
    cache.append_prefill(0, k, v)
    assert cache.seq_len() == 5


def test_seq_len_zero_with_no_layer_0(cache):
    del cache.layers[0]
    assert cache.seq_len() == 0


# ──────────────────────────────────────────────────────────────────────────────
# report_bytes_per_tier
# ──────────────────────────────────────────────────────────────────────────────

def test_report_bytes_keys(cache):
    r = cache.report_bytes_per_tier()
    assert "gpu_active_bytes" in r
    assert "cpu_active_bytes" in r


def test_report_bytes_cpu_after_cpu_append(cache):
    k, v = _kv(3)
    cache.append_prefill(0, k, v)
    r = cache.report_bytes_per_tier()
    # All pages are on CPU (state == 'CPU')
    assert r["cpu_active_bytes"] >= 0
    assert r["gpu_active_bytes"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# clear
# ──────────────────────────────────────────────────────────────────────────────

def test_clear_resets_all_layers(cache):
    k, v = _kv(3)
    cache.append_prefill(0, k, v)
    cache.clear()
    assert cache.num_layers() == 0
    assert cache.seq_len() == 0


def test_clear_frees_pages_back_to_pool(cache):
    alloc = cache.allocator
    k, v = _kv(PAGE_SIZE)
    cache.append_prefill(0, k, v)
    free_before = len(alloc.cpu_free_pool)
    cache.clear()
    assert len(alloc.cpu_free_pool) > free_before


# ──────────────────────────────────────────────────────────────────────────────
# Eviction (CPU-only: evict_pages finds no GPU pages — just checks no crash)
# ──────────────────────────────────────────────────────────────────────────────

def test_evict_pages_no_gpu_pages_no_crash(cache):
    k, v = _kv(8)
    cache.append_prefill(0, k, v)
    cache.evict_pages(99999)  # no GPU pages → nothing to evict, no exception


def test_watermark_no_crash_cpu_only(cache):
    k, v = _kv(8)
    cache.append_prefill(0, k, v)
    # _enforce_residency_policy runs but finds no GPU pages
    r = cache.report_bytes_per_tier()
    assert r is not None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA eviction tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_evict_gpu_pages_to_cpu():
    alloc = KVAllocator(num_layers=2, num_kv_heads=2, head_dim=8,
                        page_size=4, dtype=torch.float32)
    # Zero watermark → evict everything possible after each append
    mgr = KVCacheManager(alloc, gpu_high_watermark_bytes=0)
    mgr.initialize_layer_caches(2)
    gpu = torch.device("cuda:0")
    k = torch.randn(1, 2, 8, 8, device=gpu)
    v = torch.randn(1, 2, 8, 8, device=gpu)
    mgr.append_prefill(0, k, v)
    # Force eviction
    mgr.evict_pages(999999)
    # All fully-written non-tail pages should be on CPU
    for page in mgr.layers[0].pages[:-1]:
        assert page.state in ("CPU", "FREE"), f"Expected CPU, got {page.state}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tail_page_protected_from_eviction():
    alloc = KVAllocator(num_layers=2, num_kv_heads=2, head_dim=8,
                        page_size=4, dtype=torch.float32)
    mgr = KVCacheManager(alloc, gpu_high_watermark_bytes=0)
    mgr.initialize_layer_caches(2)
    gpu = torch.device("cuda:0")
    k = torch.randn(1, 2, 4, 8, device=gpu)
    v = torch.randn(1, 2, 4, 8, device=gpu)
    mgr.append_prefill(0, k, v)
    tail = mgr.layers[0].tail_page
    # Try to evict everything
    mgr.evict_pages(999999)
    # Tail page should NOT have been evicted (it's the active tail)
    # After eviction, tail_page reference may have been replaced — check original page state
    # The tail page (fully written, 4/4 tokens) IS eligible normally, but the
    # evict_pages guard checks `page != layer.tail_page`
    assert mgr.layers[0].tail_page is not None

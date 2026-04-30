import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import trange


def torch_loop_centroid_update_euclid(
    x: torch.Tensor,
    w: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
):
    """Reference Python implementation (double for-loop)"""
    B, N, D = x.shape
    K = old_centroids.shape[1]
    new_centroids = torch.zeros_like(old_centroids)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            if mask.any():
                cluster_w = w[b][mask]
                weighted_sum = torch.sum(
                    x[b][mask] * cluster_w[:, None], dim=0
                )
                w_sum = torch.sum(cluster_w, dim=0)
                new_centroids[b, k] = weighted_sum / w_sum
            else:
                new_centroids[b, k] = old_centroids[b, k]
    return new_centroids


# def triton_centroid_update_euclid(
#     x: torch.Tensor, w: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor
# ):
#     """Compute centroids for Euclidean KMeans using Triton.

#     Args:
#         x (Tensor): (B, N, D) input vectors (float16/float32)
#         cluster_ids (LongTensor): (B, N) cluster assignment per point
#         old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x)

#     Returns:
#         Tensor: (B, K, D) updated centroids (dtype == x.dtype)
#     """
#     assert (
#         x.is_cuda and cluster_ids.is_cuda
#     ), "Input tensors must be on CUDA device"
#     B, N, D = x.shape
#     K = old_centroids.shape[1]
#     assert cluster_ids.shape == (B, N)

#     # Allocate accumulation buffers
#     centroid_weighted_sums = torch.zeros(
#         (B, K, D), device=x.device, dtype=torch.float32
#     )
#     centroid_weights = torch.zeros((B, K), device=x.device, dtype=torch.float32)

#     total_tokens = B * N
#     BLOCK_D = 128  # tuneable
#     grid = (total_tokens,)

#     _centroid_update_kernel[grid](
#         x,
#         cluster_ids.to(torch.int32),
#         centroid_weighted_sums,
#         centroid_weights,
#         x.stride(0),
#         x.stride(1),
#         x.stride(2),
#         centroid_weighted_sums.stride(0),
#         centroid_weighted_sums.stride(1),
#         centroid_weighted_sums.stride(2),
#         centroid_weights.stride(0),
#         centroid_weights.stride(1),
#         B,
#         N,
#         D,
#         K,
#         BLOCK_D=BLOCK_D,
#     )

#     # Compute means; keep old centroid if empty cluster
#     counts_f = centroid_weights.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
#     centroids = centroid_weighted_sums / counts_f

#     # For clusters with zero count, revert to old centroids
#     zero_mask = (centroid_weights == 0).unsqueeze(-1)
#     centroids = torch.where(
#         zero_mask, old_centroids.to(torch.float32), centroids
#     )

#     return centroids.to(x.dtype)


# ------------------------------ NEW: chunk-wise centroid update (sorted ids) ------------------------------


@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,  # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
    w_ptr,  # *f16 / *f32 [B, N] - ORIGINAL ORDER
    sorted_idx_ptr,  # *i32        [B, N]    – indices after sort
    sorted_cluster_ptr,  # *i32        [B, N]    – cluster ids in sorted order
    wx_sum_ptr,  # *f32        [B, K, D]
    w_sum_ptr,  # *f32        [B, K]
    # strides
    stride_x_b,
    stride_x_n,
    stride_x_d,
    stride_w_b,
    stride_w_n,
    stride_idx_b,
    stride_idx_n,
    stride_cluster_b,
    stride_cluster_n,
    stride_sum_b,
    stride_sum_k,
    stride_sum_d,
    stride_count_b,
    stride_count_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,  # how many tokens (points) each program processes
):
    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.

    Because the tokens are sorted by cluster id, identical ids appear in
    contiguous runs.  We therefore accumulate a local sum/count for the
    current run and perform **a single atomic update per run**, instead of
    per-token.
    """
    # program indices – 2-D launch grid: (chunk_id, batch_id)
    pid_chunk = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    b = pid_b.to(tl.int64)
    chunk_start = (pid_chunk * BLOCK_N).to(
        tl.int64
    )  # position of the first token handled by this program

    # Nothing to do – out of range
    if chunk_start >= N:
        return

    # base pointers for this batch
    idx_batch_base = sorted_idx_ptr + b * stride_idx_b
    cid_batch_base = sorted_cluster_ptr + b * stride_cluster_b
    x_batch_base = x_ptr + b * stride_x_b  # for pointer arithmetic
    w_batch_base = w_ptr + b * stride_w_b

    # helper aranges
    offs_token = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_dim = tl.arange(0, D).to(tl.int64)

    # first token index & validity mask
    token_idx = chunk_start + offs_token
    valid_tok = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(
        cid_batch_base + token_idx * stride_cluster_n, mask=valid_tok, other=-1
    )

    all_tokens_idxs = tl.load(
        idx_batch_base + token_idx * stride_idx_n, mask=valid_tok, other=-1
    )  # [BLOCK_N]
    all_tokens_idxs = all_tokens_idxs.to(tl.int64)

    load_mask = all_tokens_idxs[:, None] * D + offs_dim[None, :]

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            row_ptrs_x = (
                x_batch_base
                + all_tokens_idxs[:, None] * stride_x_n
                + offs_dim[None, :] * stride_x_d
            )
            cluster_feats = tl.load(
                row_ptrs_x, mask=cluster_mask[:, None], other=0.0
            )  # [BLOCK_N, D]
            cluster_feats = cluster_feats.to(tl.float32)

            row_ptrs_w = w_batch_base + all_tokens_idxs * stride_w_n
            cluster_weights = tl.load(row_ptrs_w, mask=cluster_mask, other=0.0)
            cluster_weights = cluster_weights.to(tl.float32)

            sum_feats = tl.sum(
                cluster_feats * cluster_weights[:, None], axis=0
            )

            dest_ptr = (
                wx_sum_ptr
                + b * stride_sum_b
                + cid * stride_sum_k
                + offs_dim * stride_sum_d
            )
            tl.atomic_add(dest_ptr, sum_feats)

            sum_weights = tl.sum(cluster_weights, axis=0)
            tl.atomic_add(
                w_sum_ptr + b * stride_count_b + cid * stride_count_k,
                sum_weights,
            )


# ---------------------------------------------------------------------------------------------


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor,
    w: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    BLOCK_N: int = 256,
    centroid_weighted_sums: torch.Tensor = None,
    centroid_weights: torch.Tensor = None,
    calculate_new: bool = True,
):
    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors (no normalization assumed).
    w : Tensor [B, N]
        Weight vectors.
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    old_centroids : Tensor [B, K, D]
        Previous centroids (used to fill empty clusters).
    BLOCK_N : int, optional
        Tokens per Triton program (affects occupancy/perf).
    centroid_sums : Tensor [B, K, D], optional
        Pre-allocated accumulation buffer for sums.  If None, a new buffer is created.
    centroid_cnts : Tensor [B, K], optional
        Pre-allocated accumulation buffer for counts.  If None, a new buffer is created.
    calculate_new : bool, default=True
        If True, compute and return the new centroids.  If False, only update the
        accumulation buffers.

    Returns
    _________
        centroids_new : Tensor [B, K, D] or None
            Updated centroids if `calculate_new` is True; otherwise None.
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]

    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    if centroid_weighted_sums is None:
        centroid_weighted_sums = torch.zeros(
            (B, K, D), device=x.device, dtype=torch.float32
        )
    else:
        assert centroid_weighted_sums.shape == (B, K, D)

    if centroid_weights is None:
        centroid_weights = torch.zeros(
            (B, K), device=x.device, dtype=torch.float32
        )
    else:
        assert centroid_weights.shape == (B, K)

    grid = (triton.cdiv(N, BLOCK_N), B)
    _centroid_update_chunk_kernel[grid](
        x,  # original features
        w,  # weights
        sorted_idx_int,  # gather indices
        sorted_cluster_ids.to(torch.int32),
        centroid_weighted_sums,
        centroid_weights,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        sorted_idx_int.stride(0),
        sorted_idx_int.stride(1),
        sorted_cluster_ids.stride(0),
        sorted_cluster_ids.stride(1),
        centroid_weighted_sums.stride(0),
        centroid_weighted_sums.stride(1),
        centroid_weighted_sums.stride(2),
        centroid_weights.stride(0),
        centroid_weights.stride(1),
        B,
        N,
        D,
        K,
        BLOCK_N=BLOCK_N,
    )

    if calculate_new:
        # Convert sums to means; replace empty clusters with old centroids
        centroids = centroid_weighted_sums / centroid_weights.unsqueeze(-1)
        empty_mask = (centroid_weights == 0.0).unsqueeze(-1)
        centroids = torch.where(
            empty_mask, old_centroids.to(torch.float32), centroids
        )
        return centroids.to(x.dtype), centroid_weights.to(x.dtype)
    else:
        return None, None


# ------------------------------ END new implementation ------------------------------


def main():
    torch.manual_seed(0)

    B, N, D = 32, 74256, 128  # modest sizes for quick correctness test
    K = 1000
    dtype = torch.float16

    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    w = torch.rand(B, N, device="cuda", dtype=dtype)

    cluster_ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)

    # Random old centroids for handling empty clusters
    old_centroids = torch.randn(B, K, D, device="cuda", dtype=dtype)

    # ---------------- Correctness check (compile Triton kernel) ----------------
    ref_centroids = torch_loop_centroid_update_euclid(
        x, w, cluster_ids, old_centroids
    )
    # tri_centroids = triton_centroid_update_euclid(
    #     x, cluster_ids, old_centroids
    # )  # this call triggers compilation
    tri_sorted_centroids, _ = triton_centroid_update_sorted_euclid(
        x, w, cluster_ids, old_centroids
    )  # this call triggers compilation

    # # Validate correctness (includes first-run compile)
    # if torch.allclose(ref_centroids, tri_centroids, atol=1e-3, rtol=1e-3):
    #     print("Centroid update: PASS ✅")
    # else:
    #     max_diff = (ref_centroids - tri_centroids).abs().max().item()
    #     print(f"Centroid update: FAIL ❌ | max diff = {max_diff}")

    # Validate new sorted kernel
    if torch.allclose(
        ref_centroids, tri_sorted_centroids, atol=1e-3, rtol=1e-3
    ):
        print("Sorted centroid update: PASS ✅")
    else:
        max_diff = (ref_centroids - tri_sorted_centroids).abs().max().item()
        print(f"Sorted centroid update: FAIL ❌ | max diff = {max_diff}")

    # show some examples
    print(f"ref_centroids[0,0:5,0:5]: {ref_centroids[0,0:5,0:5]}")
    # print(f"tri_centroids[0,0:5,0:5]: {tri_centroids[0,0:5,0:5]}")
    print(
        f"tri_sorted_centroids[0,0:5,0:5]: {tri_sorted_centroids[0,0:5,0:5]}"
    )

    # ---------------- Efficiency benchmark (exclude compile) ----------------
    repeats = 20

    # Torch loop timing
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in trange(repeats):
        torch_loop_centroid_update_euclid(x, w, cluster_ids, old_centroids)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / repeats  # average per run (ms)

    # # Triton timing (already compiled)
    # torch.cuda.synchronize()
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    # for _ in trange(repeats):
    #     triton_centroid_update_euclid(x, cluster_ids, old_centroids)
    # end.record()
    # torch.cuda.synchronize()
    # triton_time = start.elapsed_time(end) / repeats  # average per run (ms)

    # Sorted Triton timing (already compiled)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in trange(repeats):
        triton_centroid_update_sorted_euclid(x, w, cluster_ids, old_centroids)
    end.record()
    torch.cuda.synchronize()
    triton_sorted_time = (
        start.elapsed_time(end) / repeats
    )  # average per run (ms)

    print(
        f"\n=== Efficiency (average over {repeats} runs, exclude compile) ==="
    )
    print(f"Torch loop   : {torch_time:.2f} ms")
    # print(
    #     f"Triton kernel: {triton_time:.2f} ms (speed-up x{torch_time / triton_time:.2f})"
    # )
    print(
        f"Triton sorted: {triton_sorted_time:.2f} ms (speed-up x{torch_time / triton_sorted_time:.2f})"
    )


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from tqdm import trange

from flash_kmeans.assign_euclid_triton import (  # cosine_assign_triton,
    euclid_assign_triton,
)
from flash_kmeans.centroid_update_triton import (  # triton_centroid_update_cosine,; triton_centroid_update_euclid,; triton_centroid_update_sorted_cosine,
    triton_centroid_update_sorted_euclid,
)

# -------------------- Compiled single-iteration kernels --------------------


# 1. Euclidean
def _euclid_iter(x, x_sq, w, centroids, use_heuristic=True):

    cluster_ids = euclid_assign_triton(
        x, centroids, x_sq, use_heuristic=use_heuristic
    )
    centroids_new = triton_centroid_update_sorted_euclid(
        x, w, cluster_ids, centroids
    )

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids


COMPILE_FLAG = True

try:
    if COMPILE_FLAG:
        _euclid_iter_compiled = torch.compile(
            _euclid_iter, dynamic=True, mode="reduce-overhead"
        )
    else:
        _euclid_iter_compiled = _euclid_iter
except Exception:  # pragma: no cover
    _euclid_iter_compiled = _euclid_iter


def batch_kmeans_Euclid(
    x,
    w,
    n_clusters,
    max_iters=100,
    tol=0.0,
    init_centroids=None,
    verbose=False,
    *,
    use_heuristic=True,
):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        w: Tensor of shape (B, N), batch_size B, N points per batch.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
        use_heuristic: Use heuristic Triton config (skip autotune).
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
    """
    B, N, D = x.shape

    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x**2).sum(dim=-1)  # (B, N)

    if init_centroids is None:
        # Randomly select initial centers from x
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x, dim=1, index=indices[..., None].expand(-1, -1, D)
        )  # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _euclid_iter_compiled(
            x, x_sq, w, centroids, use_heuristic
        )

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


if __name__ == "__main__":
    torch.manual_seed(0)

    # 用法示例
    B, N, D = 32, 74256, 128  # 32 个 batch，每个 batch 10 万点，128 维
    dtype = torch.float16
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)  # 大 batch 用 GPU 跑
    w = torch.rand(B, N, device="cuda", dtype=dtype)  # 大 batch 用 GPU 跑
    n_clusters = 1000
    max_iters = 2

    print("=== Testing Euclidean Distance K-Means ===")
    cluster_ids_euclid, centroids_euclid, n_iters_euclid = batch_kmeans_Euclid(
        x, w, n_clusters, max_iters=max_iters, verbose=True
    )
    print(
        f"Euclidean - cluster_ids shape: {cluster_ids_euclid.shape}, centroids shape: {centroids_euclid.shape}"
    )

    # Profile the time cost with rounds=100
    rounds = 200
    import time

    print(f"\n=== Speed Comparison (averaged over {rounds} rounds) ===")

    # Test Euclidean Distance K-Means
    euclid_start = torch.cuda.Event(enable_timing=True)
    euclid_end = torch.cuda.Event(enable_timing=True)
    euclid_start.record()
    for i in range(rounds):
        cluster_ids_euclid, centroids_euclid, n_iters_euclid = (
            batch_kmeans_Euclid(
                x,
                w,
                n_clusters,
                init_centroids=centroids_euclid,
                max_iters=max_iters,
                verbose=False,
            )
        )
    euclid_end.record()
    torch.cuda.synchronize()
    euclid_time = euclid_start.elapsed_time(euclid_end) / rounds
    euclid_time_per_iter = euclid_time / n_iters_euclid
    print(
        f"Euclidean Distance K-Means: {euclid_time:.2f} ms per run, total {n_iters_euclid} iterations, {euclid_time_per_iter:.2f} ms per iter"
    )
    print(
        f"Euclidean Distance TFLOPS: {2 * B * N * D * n_clusters * n_iters_euclid / euclid_time / 1e12:.2f}"
    )

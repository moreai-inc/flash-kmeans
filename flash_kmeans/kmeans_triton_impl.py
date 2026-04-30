import torch

from flash_kmeans.assign_euclid_triton import (  # cosine_assign_triton,
    euclid_assign_triton,
)
from flash_kmeans.centroid_update_triton import (  # triton_centroid_update_cosine,; triton_centroid_update_euclid,; triton_centroid_update_sorted_cosine,
    triton_centroid_update_sorted_euclid,
)

# -------------------- Compiled single-iteration kernels --------------------


# 1. Euclidean
def _euclid_iter(x, x_sq, w, centroids, use_heuristic=True):

    cluster_ids, cluster_dists = euclid_assign_triton(
        x, centroids, x_sq, use_heuristic=use_heuristic
    )
    centroids_new, centroid_weights = triton_centroid_update_sorted_euclid(
        x, w, cluster_ids, centroids
    )

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, centroid_weights, shift, cluster_ids, cluster_dists


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
        (
            centroids_new,
            centroid_weights,
            center_shift,
            cluster_ids,
            cluster_dists,
        ) = _euclid_iter_compiled(x, x_sq, w, centroids, use_heuristic)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, cluster_dists, centroids, centroid_weights, it + 1


if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, D = 32, 74256, 128
    dtype = torch.float16
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    # HACK for verification with sample weights as sklearn is bugged
    w = torch.randint(1, 5, (B, N), device="cuda", dtype=dtype)
    K = 1000
    max_iters = 1

    old_centroids = torch.randn(B, K, D, device="cuda", dtype=dtype)

    print("=== Testing Euclidean Distance K-Means ===")
    (
        cluster_ids_euclid,
        cluster_dists_sq_euclid,
        centroids_euclid,
        centroid_weights,
        n_iters_euclid,
    ) = batch_kmeans_Euclid(
        x,
        w,
        K,
        max_iters=max_iters,
        verbose=True,
        init_centroids=old_centroids,
    )
    print(
        f"Euclidean - cluster_ids shape: {cluster_ids_euclid.shape}, centroids shape: {centroids_euclid.shape}"
    )

    import numpy as np
    from scipy.cluster.vq import kmeans2

    for b in range(B):
        # HACK for verification with sample weights as sklearn is bugged
        original_idxs = torch.repeat_interleave(
            torch.arange(N, device="cuda", dtype=torch.long),
            w[b].to(torch.long),
            dim=0,
        ).numpy(force=True)
        X = (
            torch.repeat_interleave(x[b], w[b].to(torch.long), dim=0)
            .to(torch.float32)
            .numpy(force=True)
        )
        centers = old_centroids[b].to(torch.float32).numpy(force=True)
        scipy_clusters, scipy_labels = kmeans2(X, centers, 1, minit="matrix")

        triton_scipy_dist = np.linalg.norm(
            centroids_euclid[b].numpy(force=True) - scipy_clusters,
            axis=1,
        )
        argmax = np.argmax(triton_scipy_dist)

        scipy_original_idxs_in_cluster = np.unique(
            original_idxs[scipy_labels == argmax]
        )
        triton_original_idxs_in_cluster = torch.where(
            cluster_ids_euclid[b] == argmax
        )[0]

        if triton_original_idxs_in_cluster.shape[
            0
        ] == scipy_original_idxs_in_cluster.shape[0] and np.all(
            triton_original_idxs_in_cluster.numpy(force=True)
            == scipy_original_idxs_in_cluster
        ):
            assert triton_scipy_dist[argmax] < 5.0e-3
        else:
            triton_idxs = set(
                triton_original_idxs_in_cluster.numpy(force=True)
            )
            scipy_idxs = set(scipy_original_idxs_in_cluster)
            for triton_query_idx in triton_idxs ^ scipy_idxs:
                scipy_query_idx = np.where(original_idxs == triton_query_idx)[
                    0
                ][0]

                triton_query = x[b, triton_query_idx]
                scipy_query = X[scipy_query_idx]

                triton_query_cluster_idx = cluster_ids_euclid[
                    b, triton_query_idx
                ]
                scipy_query_cluster_idx = scipy_labels[scipy_query_idx]

                triton_cluster = old_centroids[b, triton_query_cluster_idx]
                triton_cluster_of_scipy_assignment = old_centroids[
                    b, scipy_query_cluster_idx
                ]
                scipy_cluster = centers[scipy_query_cluster_idx]
                scipy_cluster_of_triton_assignment = centers[
                    triton_query_cluster_idx
                ]

                assert (
                    abs(
                        torch.sqrt(
                            cluster_dists_sq_euclid[b, triton_query_idx]
                        ).item()
                        - torch.linalg.norm(
                            triton_query - triton_cluster_of_scipy_assignment
                        ).item()
                    )
                    < 5.0e-3
                )

                assert (
                    abs(
                        np.linalg.norm(scipy_query - scipy_cluster)
                        - np.linalg.norm(
                            scipy_query - scipy_cluster_of_triton_assignment
                        )
                    )
                    < 5.0e-3
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
        (
            cluster_ids_euclid,
            cluster_dists_sq_euclid,
            centroids_euclid,
            centroid_weights,
            n_iters_euclid,
        ) = batch_kmeans_Euclid(
            x,
            w,
            K,
            init_centroids=centroids_euclid,
            max_iters=max_iters,
            verbose=False,
        )
    euclid_end.record()
    torch.cuda.synchronize()
    euclid_time = euclid_start.elapsed_time(euclid_end) / rounds
    euclid_time_per_iter = euclid_time / n_iters_euclid
    print(
        f"Euclidean Distance K-Means: {euclid_time:.2f} ms per run, total {n_iters_euclid} iterations, {euclid_time_per_iter:.2f} ms per iter"
    )
    print(
        f"Euclidean Distance TFLOPS: {2 * B * N * D * K * n_iters_euclid / euclid_time / 1e12:.2f}"
    )

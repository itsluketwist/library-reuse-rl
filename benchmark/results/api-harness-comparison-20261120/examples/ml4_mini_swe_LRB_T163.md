# ml4_mini_swe_LRB_T163

````python
"""Cluster matched pairs of feature values with k-means (k=2)."""

import random


def _lloyd(points, centers, max_iter=300):
    """Run Lloyd's algorithm from the given initial centers.

    Returns (labels, centers, inertia).
    """
    n = len(points)
    k = len(centers)
    centers = [list(c) for c in centers]
    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        # Assignment step (ties broken towards the lower cluster index).
        for i, p in enumerate(points):
            best_j, best_d = 0, None
            for j, c in enumerate(centers):
                d = (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
                if best_d is None or d < best_d:
                    best_j, best_d = j, d
            if labels[i] != best_j:
                labels[i] = best_j
                changed = True
        # Update step.
        sums = [[0.0, 0.0] for _ in range(k)]
        counts = [0] * k
        for i, p in enumerate(points):
            j = labels[i]
            sums[j][0] += p[0]
            sums[j][1] += p[1]
            counts[j] += 1
        new_centers = []
        for j in range(k):
            if counts[j] > 0:
                new_centers.append([sums[j][0] / counts[j], sums[j][1] / counts[j]])
            else:
                # Keep an empty cluster where it was.
                new_centers.append(list(centers[j]))
        if not changed and new_centers == centers:
            centers = new_centers
            break
        centers = new_centers
    inertia = 0.0
    for i, p in enumerate(points):
        c = centers[labels[i]]
        inertia += (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2
    return labels, centers, inertia


def _initializations(points):
    """Deterministic set of candidate initial centers for k=2."""
    n = len(points)
    inits = []
    if n == 0:
        return inits
    if n == 1:
        return [[list(points[0]), list(points[0])]]
    # Pair of distinct points (first two, and extremes along each axis).
    inits.append([list(points[0]), list(points[1])])
    if n > 2:
        xs = sorted(points, key=lambda p: p[0])
        inits.append([list(xs[0]), list(xs[-1])])
        ys = sorted(points, key=lambda p: p[1])
        inits.append([list(ys[0]), list(ys[-1])])
        # Farthest-apart pair (deterministic scan, only for small n).
        best_d, best_pair = -1.0, None
        for i in range(n if n <= 800 else 800):
            for j in range(i + 1, (n if n <= 800 else 800)):
                d = (points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2
                if d > best_d:
                    best_d, best_pair = d, (points[i], points[j])
        if best_pair is not None:
            inits.append([list(best_pair[0]), list(best_pair[1])])
    # k-means++ style seeds plus random restarts from a fixed RNG.
    rng = random.Random(0)
    for _ in range(20):
        inits.append([list(points[rng.randrange(n)]), list(points[rng.randrange(n)])])
    # Deduplicate.
    seen = set()
    unique = []
    for init in inits:
        key = (tuple(init[0]), tuple(init[1]))
        if key not in seen:
            seen.add(key)
            unique.append(init)
    return unique


def _kmeans2(points):
    """Best (lowest inertia) 2-means clustering of `points`.

    Returns a list of raw cluster labels (0/1) of len(points).
    """
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return [0]
    best_labels, best_inertia = None, None
    for init in _initializations(points):
        labels, _centers, inertia = _lloyd(points, init)
        if best_inertia is None or inertia < best_inertia - 1e-12:
            best_inertia, best_labels = inertia, labels
    # Prefer sklearn's answer when it is at least as good (uses the library
    # implementation when available).
    try:
        import warnings
        import numpy as np
        from sklearn.cluster import KMeans
        X = np.asarray(points, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        sk_labels = [int(v) for v in km.labels_]
        sk_inertia = float(km.inertia_)
        if best_inertia is None or sk_inertia <= best_inertia + 1e-9:
            best_labels, best_inertia = sk_labels, sk_inertia
    except Exception:
        pass
    return best_labels


def cluster_matched_pairs(first, second):
    """Cluster the ids shared by `first` and `second` into two groups.

    For every id present in both dicts a 2D point (first[id], second[id]) is
    formed, in the order the ids appear in `first`.  The points are split into
    two clusters with k-means (k=2) and the cluster labels are returned as a
    list of ints (0 for the cluster of the first point, 1 for the other one).

    Raises ValueError if the two dicts share no id.
    """
    ids = [k for k in first if k in second]
    if not ids:
        raise ValueError("no id is present in both dicts")
    points = [(float(first[k]), float(second[k])) for k in ids]
    labels = _kmeans2(points)
    # Relabel so that the cluster of the first point is 0.
    first_label = labels[0]
    return [0 if lab == first_label else 1 for lab in labels]
````

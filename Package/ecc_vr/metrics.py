import numpy as np

def pairwise_dist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis = 2)

def weighted_distance_matrix_kde(X, h=0.15, d_manifold=1, sym_rule="min"):
    """
    Build a density-weighted distance matrix D_w such that running standard
    Vietoris–Rips on D_w is equivalent to a density-weighted filtration.

    Parameters
    ----------
    X : array, shape (n, m)
    h : float
        KDE bandwidth.
    d_manifold : int
        Intrinsic dimension used in fhat and the density scaling.
    sym_rule : {'min', 'max', 'mean'}
        How to symmetrize the per-edge density scaling:
        'min'  → divide by max(s_i, s_j)  (conservative)
        'max'  → divide by min(s_i, s_j)  (aggressive)
        'mean' → harmonic mean of s_i, s_j

    Returns
    -------
    D_w : array, shape (n, n)
        Weighted distance matrix.
    f : array, shape (n,)
        KDE values at each point.
    """
    X = np.asarray(X, dtype=float)

    D = pairwise_dist(X)

    f = fhat(X, h=h, d_manifold=d_manifold)
    f = np.maximum(f, 1e-12) # guard against zero density
    f = f / f.mean()            
    s = f ** (1.0 / d_manifold)         # (n,)

    si = s[:, None]
    sj = s[None, :]

    rule = sym_rule.lower().strip()
    if rule == "min":
        scale = np.maximum(si, sj)
    elif rule == "max":
        scale = np.minimum(si, sj)
    elif rule in {"mean", "avg", "average"}:
        scale = (si * sj) / (si + sj)
    else:
        raise ValueError("sym_rule must be 'min', 'max', or 'mean'")
    D_w = D * scale
    np.fill_diagonal(D_w, 0.0)
    return D_w, scale

def intersect_two_pointer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    #Return the intersection of two sorted 1D integer arrays using two pointers.
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    if a.size == 0 or b.size == 0:
        return np.empty(0, dtype=np.int32)
    i = 0
    j = 0
    out = []
    while i < a.size and j < b.size: #continue through each set until we hit the end of one
        ai = a[i]
        bj = b[j]

        if ai == bj: #if elements match the belong in the intersection and should be appended
            out.append(ai)
            i += 1 #move both pointers forward
            j += 1
        elif ai < bj:
            i += 1 #since arrays are sorted any ai < bj will neber match it or anything after it so advance pointer i
        else: #if b[j] is smaller it cant match anything in a at this position so move pointer j
            j += 1
    return np.array(out, dtype=np.int32)

def random_orthonormal_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))  # random matrix
    Q, _ = np.linalg.qr(A)       # Q is orthonormal (random rotation matrix)
    return Q

def fhat(X, h=0.15, d_manifold=1):
    """
    Kernel Density Estimate at each sample point using a Gaussian kernel.

    Parameters
    ----------
    X : array, shape (n, m)
        Input points.
    h : float
        Bandwidth parameter.
    d_manifold : int
        Intrinsic dimension of the manifold; used to rescale the KDE.

    Returns
    -------
    fhat : array, shape (n,)
        Estimated density at each point.

    Notes
    -----
    Internally builds an (n, n, m) array via broadcasting, so memory usage
    scales as O(n^2 * m). For large n, consider chunking or scipy.spatial.distance.cdist.
    """
    X = np.asarray(X, dtype=float)
    n, m = X.shape

    # Pairwise differences normalized by h: u_ij = (x_j - x_i) / h
    diff = (X[None, :, :] - X[:, None, :]) / h   # (n, n, m)
    r2 = np.sum(diff ** 2, axis=2)                # (n, n)

    # Gaussian kernel in R^m
    K = np.exp(-0.5 * r2) / ((2 * np.pi) ** (m / 2))  # (n, n)

    # KDE at each x_i
    f = K.sum(axis=1) / (n * (h ** d_manifold))   # (n,)
    return f
import numpy as np
from metrics import pairwise_dist

def Nplus(D: np.ndarray, epsilon: float) -> list[np.ndarray]:
    #precompute Nplus[i] = sorted array of neighbors j > i with D[i,j] <= epsilon
    n = D.shape[0] #number of vertices
    N = []
    for i in range(n):
        js = np.where(D[i, i + 1:] <= epsilon)[0] + (i + 1)
        N.append(js.astype(np.int32))
    return N

def order_vertices_by_eps_neighbors(
    X: np.ndarray,
    epsilon: float,
    D: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a permutation that orders vertices by increasing epsilon-neighbor count.

    Parameters
    ----------
    X : array, shape (n, m)
    epsilon : float
    D : array, shape (n, n), optional
        Precomputed distance matrix. Computed from X if not provided.

    Returns
    -------
    perm : array, shape (n,)
        perm[k] is the OLD index of the vertex placed at position k
        (sorted by increasing neighbor count).
    inv_perm : array, shape (n,)
        inv_perm[i] is the NEW index of old vertex i.
    """
    X = np.asarray(X, dtype=float)
    if D is None:
        D = pairwise_dist(X)

    # Degree in the epsilon-graph (exclude self)
    deg = (D <= epsilon).sum(axis=1) - 1

    # Stable sort so ties preserve original order
    perm = np.argsort(deg, kind="stable")
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return perm, inv_perm

def local_contributions_vr(
    X: np.ndarray,
    epsilon: float,
    max_dim: int | None = None,
    D: np.ndarray | None = None,
) -> list[tuple[float, int]]:
    
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if D is None:
        D = pairwise_dist(X)

    # Reorder vertices by increasing epsilon-neighbor count (optimization)
    perm, _ = order_vertices_by_eps_neighbors(X, epsilon, D=D)
    if not np.all(perm == np.arange(n)):
        D = D[np.ix_(perm, perm)]

    N = Nplus(D, epsilon)
    C: list[tuple[float, int]] = []

    def expand(simplex, candidates, f_simplex):
        dim = len(simplex) - 1
        C.append((f_simplex, 1 if dim % 2 == 0 else -1))
        if max_dim is not None and dim >= max_dim:
            return
        for v in candidates:
            v = int(v)
            f_new = max(f_simplex, D[simplex, v].max())
            simplex.append(v)
            expand(simplex, intersect_two_pointer(candidates, N[v]), f_new)
            simplex.pop()

    for i in range(n):
        expand([i], N[i], 0.0)

    C.sort(key=lambda t: t[0])
    return C

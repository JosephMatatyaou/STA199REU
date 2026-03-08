import numpy as np
from .metrics import pairwise_dist, intersect_two_pointer

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

def local_contributions(
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

def EC_from_C(C: list[tuple[float, int]]):
    # Edge case: no events
    if not C:
        return lambda r: 0

    # Split C into two arrays: filtration values and signs
    filtration_values = np.array([f for (f, _) in C], dtype=float)
    signs = np.array([s for (_, s) in C], dtype=int)

    # Prefix sums: prefix_sums[k] = sum of signs from 0 through k
    prefix_sums = np.cumsum(signs)

    def chi(r: float) -> int:
        """
        Euler characteristic at radius r:
        include all events with filtration_value <= r.
        """
        # number of events with f <= r
        k = np.searchsorted(filtration_values, r, side="right")

        # sum of first k signs
        if k == 0:
            return 0
        return int(prefix_sums[k - 1])

    return chi

def ECC_from_C(
    C: list[tuple[float, int]],
    r_min: float,
    r_max: float,
    num: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Euler Characteristic Curve (ECC) sampled on a uniform grid.

    Parameters
    ----------
    C : list of (filtration_value, sign)
        Output of local_contributions_vr, sorted by filtration value.
    r_min, r_max : float
        Range of the output grid.
    num : int
        Number of grid points.

    Returns
    -------
    r_grid : array, shape (num,)
    ecc : array, shape (num,), dtype int
    """
    if not C:
        r_grid = np.linspace(r_min, r_max, num=num)
        return r_grid, np.zeros_like(r_grid, dtype=int)

    filtration_values = np.array([f for (f, _) in C], dtype=float)
    signs = np.array([s for (_, s) in C], dtype=int)
    prefix_sums = np.cumsum(signs)

    r_grid = np.linspace(r_min, r_max, num=num)

    k = np.searchsorted(filtration_values, r_grid, side="right")
    ecc = np.zeros_like(k, dtype=int)

    mask = k > 0
    ecc[mask] = prefix_sums[k[mask] - 1]

    return r_grid, ecc

def plot_ECC_from_C(
    C: list[tuple[float, int]],
    eps_min: float,
    eps_max: float,
    num: int = 200,
    title: str = "Euler Characteristic Curve (ECC)",
    xlabel: str = "Epsilon",
    ylabel: str = "Euler Characteristic"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: compute ECC on [eps_min, eps_max] and plot it.
    Returns (r_grid, ecc).
    """
    import matplotlib.pyplot as plt
    r_grid, ecc = ECC_from_C(C, eps_min, eps_max, num=num)

    plt.figure()
    # ECC is a right-continuous step function of epsilon
    plt.step(r_grid, ecc, where="post")
    plt.xlim(eps_min, eps_max)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def chi( X: np.ndarray, eps: float, max_dim: int | None = None,) -> int:
    #Compute chi(eps) directly (no event list C).
    #Enumerates simplices in the eps-neighbor graph and accumulates (-1)^dim.
    
    X = np.asarray(X, dtype=float)
    n = X.shape[0] #number of points 

    D = pairwise_dist(X) #precompute distance matrix
    N = Nplus(D, eps) #builds neighbor lists

    # reorder vertices by increasing number of eps-neighbors (balances DFS trees)
    deg = (D <= eps).sum(axis=1) - 1
    perm = np.argsort(deg, kind="stable")
    if not np.all(perm == np.arange(n)):
        X = X[perm]
        D = D[np.ix_(perm, perm)]
        N = Nplus(D, eps)

    total = 0 #running total to store EC

    def expand(simplex: list[int], candidates: np.ndarray):
        nonlocal total

        dim = len(simplex) - 1
        sign = 1 if (dim % 2 == 0) else -1
        total += sign

        if max_dim is not None and dim >= max_dim:
            return

        for v in candidates: #loops through every possible next vertex v that can be added while staying in the complete subgraph
            v = int(v)
            candidates_new = intersect_two_pointer(candidates, N[v]) #
            simplex.append(v) #add v to the simplex
            expand(simplex, candidates_new) #explore all extensions recursively
            simplex.pop() #remove v

    for i in range(n): #start the search from each vertex
        expand([i], N[i])

    return total

def run_ec_simulation(
    n: int,
    epsilons: list[float],
    trials: int,
    max_dim: int | None = None,
    seed: int | None = None,
    X: np.ndarray | None = None,
):
    """
    Run multiple trials and compute EC at each epsilon value.
    results : dict mapping epsilon -> numpy array of EC values
    """
    rng = np.random.default_rng(seed)

    results = {eps: [] for eps in epsilons}

    if X is not None:
        X_fixed = np.asarray(X, dtype=float)
        n0 = X_fixed.shape[0]
        if n0 == 0:
            raise ValueError("Provided X is empty.")

    for _ in range(trials):
        if X is None:
            Xt = rng.uniform(0.0, 1.0, size=(n, 1))
        else:
            # bootstrap resample from the provided point cloud (same number of points each trial)
            idx = rng.choice(n0, size=n, replace=True)
            Xt = X_fixed[idx]

        for eps in epsilons:
            EC = chi(Xt, eps, max_dim=max_dim)
            results[eps].append(EC)
    for eps in epsilons:
        results[eps] = np.array(results[eps], dtype=int)

    return results


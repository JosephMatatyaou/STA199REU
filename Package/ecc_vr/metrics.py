import numpy as np

def pairwise_dist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis = 2)

def weighted_distance_matrix_kde(X, h=0.15):
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
    Returns
    -------
    D_w : array, shape (n, n)
        Weighted distance matrix.
    f : array, shape (n,)
        KDE values at each point.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    m = d - 1

    D = pairwise_dist(X)

    f = fhat(X, h=h)
    f = np.maximum(f, 1e-12) # guard against zero density         
    s = f ** (1.0 / m)         # (n,)

    si = s[:, None]
    sj = s[None, :]

    scale = (2.0 * si * sj) / (si + sj)
    
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

def fhat(X, h=0.15):
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

    n, d = X.shape

    m = d - 1

    # Pairwise differences normalized by h: u_ij = (x_j - x_i) / h

    """X[None, :, :] changes shape from (n,d) to (1,n,d)
    X[:, None, :] changes shape from (n,d) to (n,1,d)
    Numpy can then broadcast them to together creating (1,n,d) - (n,1,d) = (n,n,d)
    so each pair (i,j) diff[i,j,:] = (x_j - x_i) / h.
    So this fixes point x_i compares it to point x_j and subtracts coord by coord and then divides by h.
    diff stores every scaled difference vector between every pair of points.
    so diff is (n=first axis i,n = second axis j,d = coordinates). 
    """
    
    diff = (X[None, :, :] - X[:, None, :]) / h   # (n, n, d)
    D = np.linalg.norm(diff, axis = 2) # (n,n) axis 2 holds coordinates of the distance vector between x_i and x_j and summing across it collapses the broadcasted diff into the distance matrix divided by h

    r2 = D ** 2  

    # Gaussian kernel in R^m
    K = np.exp(-0.5 * r2) / ((2 * np.pi) ** (m / 2))  # (n, n) 

    # KDE at each x_i
    f = K.sum(axis=1) / (n * (h ** m))   # (n,)
    return f

def count_edges(X: np.ndarray, t: float, D: np.ndarray | None = None) -> int:
    X = np.asarray(X, dtype = float)

    epsilon = t/(X.shape[0]**(1/(X.shape[1]-1)))

    if D is None:
        D = pairwise_dist(X)
    
    edge_count = int(np.count_nonzero(np.triu(D <= epsilon, k = 1)))
    return edge_count
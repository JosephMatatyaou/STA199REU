# Goal: Build one file with all of our current EC functions that lets you
# choose to input discret epsilon or range of epsilons. 
# Can handle points clouds of functional data?
# Can handle points clouds of functional data?

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#helper functions

def pairwise_dist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis = 2)

def Nplus(D: np.ndarray, epsilon: float) -> list[np.ndarray]:
    #precompute Nplus[i] = sorted array of neighbors j > i with D[i,j] <= epsilon
    n = D.shape[0] #number of vertices
    N = []
    for i in range(n):
        js = np.where(D[i, i + 1:] <= epsilon)[0] + (i + 1)
        N.append(js.astype(np.int32))
    return N #prevents double counting and returns the N+ matrix for all vertices

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
    # this is how we enforce that future vertices must neighbor all current simplex vertices

# New function: order_vertices_by_eps_neighbors
def order_vertices_by_eps_neighbors(X: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      perm: length-n array of OLD indices in NEW order (increasing eps-neighbor count)
      inv_perm: length-n array mapping OLD index -> NEW index
    """
    X = np.asarray(X, dtype=float)
    D = pairwise_dist(X)

    # degree in the FULL epsilon-graph (exclude self)
    deg = (D <= epsilon).sum(axis=1) - 1

    # stable sort so ties keep original order
    perm = np.argsort(deg, kind="stable")
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(len(perm))
    return perm, inv_perm

#Big function that builds C for range of epsilons

def local_contributions_vr( #enumerating all simplices in the epsilon graph up to some max_dim
        X: np.ndarray,
        epsilon: float,
        max_dim: int | None = None,
        ) -> list[tuple[float, int]]:
    """
    Using clean depth-first search complete subgraph enumeration in the epsilon graph with ordered neighbor sets.
    Emits (filtration_value, (-1)^dim) for each simplex found to save memory.
    outputs C which is a list of events at filtration value f add 1 or subtract 1
    """
    X = np.asarray(X, dtype = float)
    n = X.shape[0]
    D = pairwise_dist(X)
    N = Nplus(D, epsilon)

    # reorder vertices by increasing number of epsilon-neighbors (balances DFS trees)
    deg = (D <= epsilon).sum(axis=1) - 1
    perm = np.argsort(deg, kind="stable")
    if not np.all(perm == np.arange(n)):
        X = X[perm]
        D = D[np.ix_(perm, perm)]
        N = Nplus(D, epsilon)

    C: list[tuple[float, int]] = [] # candidate set is a set of float,int tuples with (filtration_val, sign)
    
    def expand(simplex: list[int], candidates: np.ndarray, f_simplex: float):
        #simplex is a list of vertex indices in sigma. Candidates is the allowed vertices to add next (the intersection set).
        #f_simplex is the current filtration value for sigma (max edge in sigma so far). 

        #emit contribution of the current simplex for memory
        dim = len(simplex) - 1
        sign = 1 if (dim % 2 == 0) else -1
        C.append((f_simplex, sign))

        #dimension cap (don't expand further than max_dim)
        if max_dim is not None and dim >= max_dim:
            return
        
        #now try expanding by each candidate vertex v
        for v in candidates:
            v = int(v)
            #update filtration with max of current filtration and longest edge to v
            longest_edge = 0.0
            for u in simplex:
                d = D[u, v]
                if d > longest_edge:
                    longest_edge = d
            f_new = max(f_simplex, longest_edge)
            # computes max(f_sigma, maxD[u,v] for u in sigma)

            #update candidate set (C intersect N+[v] = C(sigma union {v}))
            candidates_new = intersect_two_pointer(candidates, N[v])

            simplex.append(v) #add v
            expand(simplex, candidates_new, f_new)
            simplex.pop()
    
    #start from each vertex i as the smallest vertex index
    for i in range(n):
        expand([i], N[i], 0.0)
    
    C.sort(key = lambda t: t[0]) #sort by filtration value to to cumulatibe sums in order
    return C

#computes the EC from C
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


# Compute an Euler Characteristic Curve (ECC) on a grid of r values

def ECC_from_C(
    C: list[tuple[float, int]],
    r_min: float,
    r_max: float,
    num: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    #Compute the Euler Characteristic Curve (ECC) from an event list C.

    # Edge case: no events
    if not C:
        r_grid = np.linspace(r_min, r_max, num=num)
        return r_grid, np.zeros_like(r_grid, dtype=int)

    # Assumes C is sorted by filtration value (local_contributions_vr already sorts)
    filtration_values = np.array([f for (f, _) in C], dtype=float)
    signs = np.array([s for (_, s) in C], dtype=int)
    prefix_sums = np.cumsum(signs)

    r_grid = np.linspace(r_min, r_max, num=num)

    # For each r, include all events with filtration_value <= r
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: compute ECC on [eps_min, eps_max] and plot it.
    Returns (r_grid, ecc).
    """
    r_grid, ecc = ECC_from_C(C, eps_min, eps_max, num=num)

    plt.figure()
    # ECC is a right-continuous step function of epsilon
    plt.step(r_grid, ecc, where="post")
    plt.xlim(eps_min, eps_max)
    plt.xlabel("epsilon (r)")
    plt.ylabel("chi(r)")
    plt.title(title)
    plt.show()

    return r_grid, ecc


#computes EC directly without C
def chi( X: np.ndarray, eps: float, max_dim: int | None = None,) -> int:
    """
    Compute chi(eps) directly (no event list C).
    Enumerates simplices in the eps-neighbor graph and accumulates (-1)^dim.
    """
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

#runs simulation for discrete epsilon values
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

    - If X is provided: each trial draws a fresh sample of n points from rows of X (with replacement).
      (So the sampled point cloud changes each trial, but the number of points stays the same.)
    - If X is None: each trial samples n points Uniform[0,1] in 1D.

    Parameters
    ----------
    n : int
        number of sampled points per trial when X is None
    epsilons : list[float]
        epsilon (r) values
    trials : int
        number of simulation runs
    max_dim : int | None
        maximum simplex dimension (None = no cap)
    seed : int | None
        RNG seed
    X : np.ndarray | None
        optional fixed point cloud to resample for every trial

    Returns
    -------
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

# Helper function: ECC for a point cloud X

def ECC_for_point_cloud(
    X: np.ndarray,
    r_min: float,
    r_max: float,
    num: int = 200,
    max_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-shot helper: build C at radius r_max, then compute/plot ECC on [r_min, r_max].
    """
    C = local_contributions_vr(X, r_max, max_dim=max_dim)
    return ECC_from_C(C, r_min, r_max, num=num)

#Example Point Clouds

def apply_noise(Y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    #add noise in ambient dim coords
    if noise <=0:
        return Y
    return Y + rng.normal(0, noise, size=Y.shape)

def unif_torus_points(n: int, R: float, r: float, seed: int) -> np.ndarray:
    """
    Torus parameterization:
    x = (R + r cos(theta)) cos(phi)
    y = (R + r cos(theta)) sin(phi)
    z = r sin(theta)

    Uniform in (theta, phi) is no uniform in area because surface area is
    dA = r(R + rcos(theta)) dtheta dphi. So we sample phi ~ Unif[0,2pi) and sample
    theta with density proportional to (R + rcos(theta)) using acceptance-rejectin.
    """
    if n <= 0:
        return np.empty((0,3))
    if R <=0 or r <= 0:
        raise ValueError("R and r must be > 0")
    #if r >= R:
     #   pass

    rng = np.random.default_rng(seed)
    out = np.empty((n,3), dtype = float)

    phi = rng.uniform(0,2 * np.pi, size = n)

    j = 0
    M = R + r #envelope for acceptance-rejection since R + rcos(theta) <= R + r
    while j < n:
        theta = rng.uniform(0,2 * np.pi)
        h = R + r * np.cos(theta)
        if rng.uniform(0,M) <= h:
            out[j, 0] = h * np.cos(phi[j])
            out[j, 1] = h * np.sin(phi[j])
            out[j, 2] = r * np.sin(theta)
            j += 1
    return out

# Plot helpers

def pca_project(X: np.ndarray, out_dim: int = 2) -> np.ndarray: #projects data into 2D using PCA via SVD
    Xc = X - X.mean(axis=0, keepdims=True) #centers coords to have mean 0
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False) #SVD of centered data
    W = Vt[:out_dim].T #takes out first out_dim principal directions and transpose into projection matrix
    return Xc @ W #multiply centered data with projection matrix to get lower-dimensional coordinates

def plot_point_cloud_on_ax(ax, X: np.ndarray, dim: int):
    if dim == 1:
        ax.scatter(X[:, 0], np.zeros_like(X[:, 0]), s=10)
        ax.set_xlabel("x")
        ax.set_yticks([])
        ax.set_title("Point Cloud (1D)")
        return

    if dim == 2:
        ax.scatter(X[:, 0], X[:, 1], s=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Point Cloud (2D)")
        ax.axis("equal")
        return

    if dim == 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Point Cloud (3D)")
        return

    # dim >= 4: show PCA 2D projection 
    X2 = pca_project(X, out_dim=2)
    ax.scatter(X2[:, 0], X2[:, 1], s=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Point cloud (PCA 2D from dim={dim})")
    ax.axis("equal")



# Top-level function: random_orthonormal_matrix
def random_orthonormal_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))  # random matrix
    Q, _ = np.linalg.qr(A)       # Q is orthonormal (random rotation matrix)
    return Q

def embed_in_ambient(
    X: np.ndarray,
    ambient_dim: int,
    seed: int,
    rotate: bool = True,
    rotate_seed: int | None = None,
) -> np.ndarray:
    
    #Embed low-dim coordinates into R^ambient_dim by padding zeros, then optionally rotate.
    n, d0 = X.shape #n is number of points, d0 is intrinsic dimension of shape
    if ambient_dim < d0:
        raise ValueError(f"Ambient dimension {ambient_dim} must be >= intrinsic embedding dim {d0}.")
    if ambient_dim == d0: #no embedding needed
        Y = X.copy()
    else: #pads with zeros to embed into higher-dimensional space
        Y = np.zeros((n, ambient_dim))
        Y[:, :d0] = X

    if rotate:  # random rotation so shape isn't aligned with coordinate axes
        seed_used = (seed + 12345) if rotate_seed is None else int(rotate_seed)
        Q = random_orthonormal_matrix(ambient_dim, seed=seed_used)
        Y = Y @ Q
    return Y

def sample_point_cloud(
    shape: str,
    n_points: int,
    ambient_dim: int,
    seed: int = 0,
    noise: float = 0.0,
    circle_radius: float = 1.0,
    cylinder_radius: float = 1.0,
    cylinder_height: float = 2.0,
    Torus_R: float = 2.0,
    Torus_r: float = 0.7,
    sphere_radius: float = 1.0,
    rotate: bool = True,
    rotate_seed: int | None = None,
) -> np.ndarray:
    """
    Supported shapes:
      - "Normal Blob" (normal point cloud) in R^ambient_dim
      - "Circle"   (S^1) embedded in R^2 then into R^ambient_dim
      - "Filled Disk" (S^1) embedded in R^2 then into R^ambient_dim
      - "Figure 8" (wedge of two circles) embedded in R^2 then into R^ambient_dim
      - "Cylinder" (S^1 x [0,1]) embedded in R^3 then into R^ambient_dim
      - "Closed Cylinder" (side + filled caps) embedded in R^3 then into R^ambient_dim
      - "Sphere" (S^2) embedded in R^3 then into R^ambient_dim
      - "Torus"    (S^1 x S^1) embedded in R^3 then into R^ambient_dim
      - "Swiss Roll" (rolled 2D manifold) embedded in R^3 then into R^ambient_dim
    """
    rng = np.random.default_rng(seed)


    #normalize names
    shape_key = shape.lower().strip()
    for ch in [" ", "-", "_", "(", ")"]:
        shape_key = shape_key.replace(ch, "")

    if shape_key in {"normalblob", "gaussian"}:
        # Gaussian "blob" in ambient space
        X = rng.normal(0.0, 1.0, size=(n_points, ambient_dim))
        X = apply_noise(X, noise, rng)
        return X

    if shape_key == "circle":
        if ambient_dim < 2:
            raise ValueError("Circle needs ambient_dim >= 2.")
        # Uniform on S^1: theta ~ Unif[0, 2pi)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        X0 = np.column_stack([circle_radius * np.cos(theta), circle_radius * np.sin(theta)])
        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y

    if shape_key in {"filleddisk", "disk", "filleddisc", "disc"}:
        if ambient_dim < 2:
            raise ValueError("Filled Disk needs ambient_dim >= 2.")
        # Uniform in area on disk: r = R*sqrt(U), theta ~ Unif[0,2pi)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        u = rng.uniform(0.0, 1.0, size=n_points)
        rad = circle_radius * np.sqrt(u)
        X0 = np.column_stack([rad * np.cos(theta), rad * np.sin(theta)])
        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y

    if shape_key in {"figure8", "figureeight"}:
        # Two circles meeting at the origin; sample uniformly on each loop
        if ambient_dim < 2:
            raise ValueError("Figure 8 needs ambient_dim >= 2.")

        n1 = n_points // 2
        n2 = n_points - n1
        # Interpret circle_radius as the radius of each loop
        Rloop = circle_radius
        c1 = np.array([-circle_radius, 0.0])
        c2 = np.array([ circle_radius, 0.0])

        th1 = rng.uniform(0.0, 2.0 * np.pi, size=n1)
        th2 = rng.uniform(0.0, 2.0 * np.pi, size=n2)
        X1 = np.column_stack([Rloop * np.cos(th1), Rloop * np.sin(th1)]) + c1
        X2 = np.column_stack([Rloop * np.cos(th2), Rloop * np.sin(th2)]) + c2
        X0 = np.vstack([X1, X2])

        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y
    
    if shape_key == "cylinder":
        if ambient_dim < 3:
            raise ValueError("Cylinder needs ambient_dim >= 3.")
        if cylinder_height <= 0:
            raise ValueError("cylinder_height must be > 0.")

        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
        z = rng.uniform(0.0, cylinder_height, size=n_points)
        X0 = np.column_stack([
            cylinder_radius * np.cos(theta),
            cylinder_radius * np.sin(theta),
            z,
        ])
        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y

    if shape_key in {"closedcylinder", "cylinderwithcaps", "cylinderfilledends", "closed"}:
        if ambient_dim < 3:
            raise ValueError("Closed Cylinder needs ambient_dim >= 3.")
        if cylinder_height <= 0:
            raise ValueError("Cylinder Height must be > 0.")

        # Mix points between side and caps
        n_side = int(round(0.6 * n_points))
        n_cap_each = (n_points - n_side) // 2
        n_top = n_cap_each
        n_bot = n_points - n_side - n_top

        # Side: uniform on surface
        theta_s = rng.uniform(0.0, 2.0 * np.pi, size=n_side)
        z_s = rng.uniform(0.0, cylinder_height, size=n_side)
        Xs = np.column_stack([
            cylinder_radius * np.cos(theta_s),
            cylinder_radius * np.sin(theta_s),
            z_s,
        ])

        # Caps: uniform in area on disks at z = 0 and z = cylinder_height
        def sample_cap(nc: int, z0: float) -> np.ndarray:
            th = rng.uniform(0.0, 2.0 * np.pi, size=nc)
            u = rng.uniform(0.0, 1.0, size=nc)
            rad = cylinder_radius * np.sqrt(u)
            x = rad * np.cos(th)
            y = rad * np.sin(th)
            z = np.full(nc, z0)
            return np.column_stack([x, y, z])

        # Caps at z = 0 and z = cylinder_height
        Xb = sample_cap(n_bot, 0.0)
        Xt = sample_cap(n_top, cylinder_height)

        X0 = np.vstack([Xs, Xt, Xb])
        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y
    
    if shape_key == "sphere":
        if ambient_dim < 3:
            raise ValueError("Sphere needs ambient_dim >= 3.")

        v = rng.normal(size=(n_points, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        X0 = sphere_radius * v
        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y
    
    if shape_key == "torus":
        if ambient_dim < 3:
            raise ValueError("Torus needs ambient_dim >= 3.")

        X0 = unif_torus_points(n_points, Torus_R, Torus_r, seed=seed)
        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y

    if shape_key == "swissroll":
        if ambient_dim < 3:
            raise ValueError("Swiss Roll needs ambient_dim >= 3")

        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n_points)
        h = rng.uniform(-1.0, 1.0, size=n_points)

        x = t * np.cos(t)
        y = h
        z = t * np.sin(t)
        X0 = np.column_stack([x, y, z])

        Y = embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
        Y = apply_noise(Y, noise, rng)
        return Y
    
    raise ValueError(f"Unknown shape: {shape}. Choose normal blob, circle, disk, figure 8, cylinder, closed cylinder sphere, torus, or swiss roll.")



# GUI

def _parse_float_list(s: str) -> list[float]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        out.append(float(p))
    return out


def _parse_int_or_none(s: str) -> int | None:
    s = (s or "").strip()
    if s == "" or s.lower() in {"none", "null"}:
        return None
    return int(s)


class ECApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("The Euler Machine")
        self.geometry("1200x800")

        # State
        self.X: np.ndarray | None = None

        # Layout
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.ctrl = ttk.Frame(self, padding=10)
        self.ctrl.grid(row=0, column=0, sticky="nsew")

        self.plot_frame = ttk.Frame(self, padding=10)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        # Matplotlib figure in Tk (left: point cloud, right: results)
        # Slightly smaller figure so the plots have more breathing room in the GUI
        self.fig = plt.Figure(figsize=(7.0, 4.4), dpi=100)
        self.ax_pc = self.fig.add_subplot(121)
        self.ax_res = self.fig.add_subplot(122)
        # Add margins + a bit more space between subplots
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92, wspace=0.30)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Controls
        self._build_controls()

    def _build_controls(self):
        r = 0

        ttk.Label(self.ctrl, text="Point cloud", font=("Helvetica", 12, "bold")).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        shapes = [
            "Normal Blob",
            "Circle",
            "Filled Disk",
            "Figure 8",
            "Cylinder",
            "Closed Cylinder",
            "Sphere",
            "Torus",
            "Swiss Roll",
        ]

        ttk.Label(self.ctrl, text="Shape").grid(row=r, column=0, sticky="w")
        self.shape_var = tk.StringVar(value="Circle")
        ttk.Combobox(self.ctrl, textvariable=self.shape_var, values=shapes, state="readonly", width=20).grid(row=r, column=1, sticky="ew")
        r += 1

        ttk.Label(self.ctrl, text="Points").grid(row=r, column=0, sticky="w")
        self.n_points_var = tk.StringVar(value="150")
        ttk.Entry(self.ctrl, textvariable=self.n_points_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Dimension").grid(row=r, column=0, sticky="w")
        self.ambient_dim_var = tk.StringVar(value="2")
        ttk.Entry(self.ctrl, textvariable=self.ambient_dim_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Seed").grid(row=r, column=0, sticky="w")
        self.seed_var = tk.StringVar(value="0")
        ttk.Entry(self.ctrl, textvariable=self.seed_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Noise (Optional)").grid(row=r, column=0, sticky="w")
        self.noise_var = tk.StringVar(value="0.0")
        ttk.Entry(self.ctrl, textvariable=self.noise_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        self.rotate_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.ctrl, text="Random Rotation", variable=self.rotate_var).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        ttk.Separator(self.ctrl).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1

        ttk.Label(self.ctrl, text="EC / ECC", font=("Helvetica", 12, "bold")).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Max Simplex Dimension").grid(row=r, column=0, sticky="w")
        self.max_dim_var = tk.StringVar(value="2")
        ttk.Entry(self.ctrl, textvariable=self.max_dim_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        # Mode
        self.mode_var = tk.StringVar(value="range")
        ttk.Radiobutton(self.ctrl, text="Epsilon Range (ECC)", value="range", variable=self.mode_var).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Radiobutton(self.ctrl, text="Discrete Epsilon List (EC)", value="discrete", variable=self.mode_var).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        # Range inputs
        ttk.Label(self.ctrl, text="Epsilon Minimum").grid(row=r, column=0, sticky="w")
        self.eps_min_var = tk.StringVar(value="0.2")
        ttk.Entry(self.ctrl, textvariable=self.eps_min_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Epsilon Maximum").grid(row=r, column=0, sticky="w")
        self.eps_max_var = tk.StringVar(value="0.7")
        ttk.Entry(self.ctrl, textvariable=self.eps_max_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Grid points (num)").grid(row=r, column=0, sticky="w")
        self.num_var = tk.StringVar(value="300")
        ttk.Entry(self.ctrl, textvariable=self.num_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        # Discrete inputs
        ttk.Label(self.ctrl, text="Epsilon list (e.g. 0.15,0.3,0.6)").grid(row=r, column=0, sticky="w")
        self.eps_list_var = tk.StringVar(value="0.15, 0.30, 0.60")
        ttk.Entry(self.ctrl, textvariable=self.eps_list_var, width=22).grid(row=r, column=1, sticky="ew")
        r += 1

        # Simulation controls (for discrete mode)
        self.sim_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.ctrl, text="Run Simulation (discrete mode only)", variable=self.sim_var).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Trials").grid(row=r, column=0, sticky="w")
        self.trials_var = tk.StringVar(value="30")
        ttk.Entry(self.ctrl, textvariable=self.trials_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Label(self.ctrl, text="Sim seed").grid(row=r, column=0, sticky="w")
        self.sim_seed_var = tk.StringVar(value="0")
        ttk.Entry(self.ctrl, textvariable=self.sim_seed_var, width=12).grid(row=r, column=1, sticky="w")
        r += 1

        ttk.Separator(self.ctrl).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1

        ttk.Button(self.ctrl, text="Generate Point Cloud", command=self._generate_point_cloud).grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        ttk.Button(self.ctrl, text="Run", command=self._run).grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        ttk.Separator(self.ctrl).grid(row=r, column=0, columnspan=2, sticky="ew", pady=8)
        r += 1

        ttk.Label(self.ctrl, text="Output").grid(row=r, column=0, sticky="w")
        r += 1

        self.out = tk.Text(self.ctrl, height=12, width=34)
        self.out.grid(row=r, column=0, columnspan=2, sticky="nsew")
        self.ctrl.rowconfigure(r, weight=1)

    def _ensure_pointcloud_axes(self, want_3d: bool):
        """Ensure the left subplot is 2D or 3D as needed."""
        is_3d = getattr(self.ax_pc, "name", "") == "3d"
        if want_3d and not is_3d:
            self.fig.delaxes(self.ax_pc)
            self.ax_pc = self.fig.add_subplot(121, projection="3d")
            # Add margins + a bit more space between subplots
            self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92, wspace=0.30)
        elif (not want_3d) and is_3d:
            self.fig.delaxes(self.ax_pc)
            self.ax_pc = self.fig.add_subplot(121)
            # Add margins + a bit more space between subplots
            self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.92, wspace=0.30)

    def _log(self, msg: str):
        self.out.insert("end", msg + "\n")
        self.out.see("end")

    def _clear_plot(self):
        self.ax_pc.clear()
        self.ax_res.clear()
        self.canvas.draw()

    def _generate_point_cloud(self):
        try:
            shape = self.shape_var.get()
            n_points = int(self.n_points_var.get())
            ambient_dim = int(self.ambient_dim_var.get())
            seed = int(self.seed_var.get())
            noise = float(self.noise_var.get())
            rotate = bool(self.rotate_var.get())

            X = sample_point_cloud(
                shape=shape,
                n_points=n_points,
                ambient_dim=ambient_dim,
                seed=seed,
                noise=noise,
                rotate=rotate,
            )
            self.X = X

            # quick visualization (point cloud on ax_pc)
            d = X.shape[1]
            self.ax_pc.clear()
            self.ax_res.clear()
            self._ensure_pointcloud_axes(want_3d=(d == 3))

            if d == 1:
                self.ax_pc.scatter(X[:, 0], np.zeros_like(X[:, 0]), s=10)
                self.ax_pc.set_yticks([])
                self.ax_pc.set_xlabel("x")
                self.ax_pc.set_title("Point Cloud")
            elif d == 2:
                self.ax_pc.scatter(X[:, 0], X[:, 1], s=10)
                self.ax_pc.set_xlabel("x")
                self.ax_pc.set_ylabel("y")
                self.ax_pc.set_title("Point Cloud")
                self.ax_pc.axis("equal")
            elif d == 3:
                self.ax_pc.scatter(X[:, 0], X[:, 1], X[:, 2], s=10)
                self.ax_pc.set_xlabel("x")
                self.ax_pc.set_ylabel("y")
                self.ax_pc.set_zlabel("z")
                self.ax_pc.set_title("Point Cloud")
            else:
                X2 = pca_project(X, out_dim=2)
                self.ax_pc.scatter(X2[:, 0], X2[:, 1], s=10)
                self.ax_pc.set_xlabel("PC1")
                self.ax_pc.set_ylabel("PC2")
                self.ax_pc.set_title(f"Point cloud (PCA 2D from dim={d})")
                self.ax_pc.axis("equal")

            self.canvas.draw()
            self._log(f"Generated {shape}: X.shape={X.shape}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run(self):
        try:
            if self.X is None:
                self._generate_point_cloud()
                if self.X is None:
                    return

            X = self.X
            mode = self.mode_var.get()
            max_dim = _parse_int_or_none(self.max_dim_var.get())

            d = X.shape[1]
            self.ax_pc.clear()
            self.ax_res.clear()
            self._ensure_pointcloud_axes(want_3d=(d == 3))

            # Always show point cloud on left
            if d == 1:
                self.ax_pc.scatter(X[:, 0], np.zeros_like(X[:, 0]), s=10)
                self.ax_pc.set_yticks([])
                self.ax_pc.set_xlabel("x")
                self.ax_pc.set_title("Point cloud")
            elif d == 2:
                self.ax_pc.scatter(X[:, 0], X[:, 1], s=10)
                self.ax_pc.set_xlabel("x")
                self.ax_pc.set_ylabel("y")
                self.ax_pc.set_title("Point Cloud")
                self.ax_pc.axis("equal")
            elif d == 3:
                self.ax_pc.scatter(X[:, 0], X[:, 1], X[:, 2], s=10)
                self.ax_pc.set_xlabel("x")
                self.ax_pc.set_ylabel("y")
                self.ax_pc.set_zlabel("z")
                self.ax_pc.set_title("Point Cloud")
            else:
                X2 = pca_project(X, out_dim=2)
                self.ax_pc.scatter(X2[:, 0], X2[:, 1], s=10)
                self.ax_pc.set_xlabel("PC1")
                self.ax_pc.set_ylabel("PC2")
                self.ax_pc.set_title(f"Point cloud (PCA 2D from dim={d})")
                self.ax_pc.axis("equal")

            # ECC/EC results on right
            if mode == "range":
                eps_min = float(self.eps_min_var.get())
                eps_max = float(self.eps_max_var.get())
                num = int(self.num_var.get())

                # Build C once at eps_max, then plot ECC over the range
                C = local_contributions_vr(X, eps_max, max_dim=max_dim)
                r_grid, ecc = ECC_from_C(C, eps_min, eps_max, num=num)

                self.ax_res.step(r_grid, ecc, where="post")
                self.ax_res.set_xlim(eps_min, eps_max)
                self.ax_res.set_xlabel("Epsilon")
                self.ax_res.set_ylabel("Euler Characteristic")
                self.ax_res.set_title("Euler Characteristic Curve (ECC)")
                self.canvas.draw()

                self._log(f"ECC computed on [{eps_min}, {eps_max}] with num={num} (max_dim={max_dim})")

            else:
                eps_list = _parse_float_list(self.eps_list_var.get())
                if not eps_list:
                    raise ValueError("Please provide a non-empty eps list.")

                # Compute EC for the current point cloud at each eps
                vals = []
                for eps in eps_list:
                    vals.append(chi(X, eps, max_dim=max_dim))

                self._log("EC values for current point cloud:")
                for eps, v in zip(eps_list, vals):
                    self._log(f"  eps={eps:.6g}: chi={v}")

                # Optional: simulation + histogram
                if self.sim_var.get():
                    trials = int(self.trials_var.get())
                    sim_seed = int(self.sim_seed_var.get())
                    results = run_ec_simulation(
                        n=X.shape[0],
                        epsilons=eps_list,
                        trials=trials,
                        max_dim=max_dim,
                        seed=sim_seed,
                        X=X,
                    )

                    # Plot histogram(s) on the right axes
                    self.ax_res.clear()
                    for eps in eps_list:
                        arr = results[eps]
                        self.ax_res.hist(arr, bins=20, alpha=0.5, label=f"eps={eps:.3g}")

                    self.ax_res.set_xlabel("EC")
                    self.ax_res.set_ylabel("count")
                    self.ax_res.set_title(f"Simulation Histogram (n={X.shape[0]}, trials={trials})")
                    self.ax_res.legend()
                    self.canvas.draw()

                    self._log("Simulation Summary:")
                    for eps in eps_list:
                        arr = results[eps]
                        self._log(
                            f"  eps={eps:.6g}: mean={arr.mean():.3f}, std={arr.std(ddof=0):.3f}, min={arr.min()}, max={arr.max()}"
                        )
                else:
                    # Just plot the EC values as a bar plot on the right
                    self.ax_res.clear()
                    self.ax_res.bar([str(eps) for eps in eps_list], vals)
                    self.ax_res.set_xlabel("Epsilon")
                    self.ax_res.set_ylabel("EC")
                    self.ax_res.set_title("EC values")
                    self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = ECApp()
    app.mainloop()

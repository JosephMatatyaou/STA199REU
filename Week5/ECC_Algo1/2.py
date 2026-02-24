#small scale example
import sys
sys.path.append(r"C:\Users\pokem\STA199REU\Week5")
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
from ECCwithVRApp import build_vr_simplex_tree, compute_ecc
# Simple test point cloud
np.random.seed(0)
X = np.random.normal(0, 1, size=(30, 2))

max_edge_length = 1.5
max_simplex_dim = 2   # vertices, edges, triangles
n_steps = 200




def build_vr_contributions(X: np.ndarray,
                           max_edge_length: float,
                           max_simplex_dim: int):
    """
    Algorithm 1 + Algorithm 2:
    Computes local Euler characteristic contributions for a
    Vietoris–Rips complex without building a simplex tree.

    Returns: list of (filtration_value, contribution)
    """
    n = X.shape[0]
    contributions = []

    # Precompute pairwise distances
    dist = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    def increase_dimension(simplex, filtration, common_neighbors):
        """
        Algorithm 2: recursively extend simplices
        """
        dim = len(simplex) - 1
        if dim >= max_simplex_dim:
            return

        for v in sorted(common_neighbors):
            # update filtration (diameter)
            new_filt = filtration
            for u in simplex:
                new_filt = max(new_filt, dist[u, v])

            if new_filt > max_edge_length:
                continue

            new_simplex = simplex + (v,)
            new_dim = len(new_simplex) - 1

            # Euler contribution
            contributions.append((new_filt, (-1) ** new_dim))

            # update common neighbors
            new_common = {
                w for w in common_neighbors
                if w > v and dist[v, w] <= max_edge_length
            }

            increase_dimension(new_simplex, new_filt, new_common)

    # Algorithm 1: loop over vertices
    for i in range(n):
        # 0-simplex contribution
        contributions.append((0.0, +1))

        # neighbors j > i within epsilon
        neighbors = {
            j for j in range(i + 1, n)
            if dist[i, j] <= max_edge_length
        }

        increase_dimension((i,), 0.0, neighbors)

    return contributions

def compute_ecc_new(contributions, n_steps: int = 250):
    """
    Computes the Euler characteristic curve from
    (filtration, ±1) contributions.
    """
    contributions.sort(key=lambda x: x[0])

    fmin = contributions[0][0]
    fmax = contributions[-1][0]
    ts = np.linspace(fmin, fmax, n_steps)

    ecc = []
    chi = 0
    idx = 0

    for t in ts:
        while idx < len(contributions) and contributions[idx][0] <= t:
            chi += contributions[idx][1]
            idx += 1
        ecc.append(chi)

    return ts, np.array(ecc)

contributions = build_vr_contributions(
    X,
    max_edge_length=max_edge_length,
    max_simplex_dim=max_simplex_dim
)


t, ecc = compute_ecc_new(contributions, n_steps=n_steps)

plt.figure(figsize=(6, 4))
plt.plot(t, ecc)
plt.xlabel("VR radius")
plt.ylabel("Euler characteristic")
plt.title("ECC via Local Contributions (Algo 1 + 2)")
plt.grid(True)
plt.show()

# --- Simplex-tree ECC (baseline) ---
st = build_vr_simplex_tree(X, max_edge_length, max_simplex_dim)
t_ref, ecc_ref = compute_ecc(st, n_steps=n_steps)

# --- Algorithmic ECC ---
contributions = build_vr_contributions(X, max_edge_length, max_simplex_dim)
t_alg, ecc_alg = compute_ecc(contributions, n_steps=n_steps)

# --- Plot comparison ---
plt.figure(figsize=(6, 4))
plt.plot(t_ref, ecc_ref, label="Simplex Tree (Gudhi)", linewidth=2)
plt.plot(t_alg, ecc_alg, "--", label="Algo 1+2 (Local)", linewidth=2)
plt.xlabel("VR radius")
plt.ylabel("Euler characteristic")
plt.title("ECC Comparison")
plt.legend()
plt.grid(True)
plt.show()
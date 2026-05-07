import numpy as np
from math import gamma
import ecc_vr as vr
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

def pairwise_dist(X: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis = 2)


# fallback so old code still works
def count_edges(X: np.ndarray, t: float, D: np.ndarray | None = None) -> int:
    X = np.asarray(X, dtype = float)

    epsilon = t/(X.shape[0]**(1/(X.shape[1]-1)))

    if D is None:
        D = pairwise_dist(X)
    
    edge_count = int(np.count_nonzero(np.triu(D <= epsilon, k = 1)))
    return edge_count

def count_edges_kde(X: np.ndarray, t: float, h: float) -> int:
    X = np.asarray(X, dtype = float)


    epsilon = t/(X.shape[0]**(1/(X.shape[1]-1)))

    D, _ = vr.weighted_distance_matrix_kde(X, h)
    
    edge_count = int(np.count_nonzero(np.triu(D <= epsilon, k = 1)))
    return edge_count

def unif_hypersphere(n_points, d, r = 1.0, center = None, seed = None):
    # generate points uniformly on the surface of an n-dimensional hypersphere
    # params: n_points (number of points), d (dimension), r (radius), center (if none uses origin)
    # returns: ndarray of shape (n_points, d)

    rng = np.random.default_rng(seed)

    #sample from standard normal and normalize
    points = rng.normal(size=(n_points,d))
    points /= np.linalg.norm(points,axis=1,keepdims = True)

    #scale to radius
    points *= r

    #shift to center
    if center is not None:
        center = np.asarray(center)
        if center.shape != (d,):
            raise ValueError(f"center must have shape ({d},)")
        points += center

    return points



def unif_hyperellipsoid(n_points, *stretch, center=None, seed=None, batch_size=500):
    """
    Uniformly sample points on the boundary of the full d-dimensional hyperellipsoid

        ((x1-c1)/a1)^2 + ((x2-c2)/a2)^2 + ... + ((xd-cd)/ad)^2 = 1

    n_samples : int
        Number of points to generate.

    *stretch : floats or one array-like
        The semi-axis lengths / stretch factors.
        Examples:
            sample_hyperellipsoid_boundary(1000, 10, 4)
            sample_hyperellipsoid_boundary(1000, 10, 4, 2, 1)
            sample_hyperellipsoid_boundary(1000, [10, 4, 2, 1])

    center : array-like of shape (d,), optional
        Center of the hyperellipsoid. Default is the origin.

    seed : int, optional
        Random seed.

    batch_size : int, optional
        Candidate points per loop. Keep this moderate for very high dimension.
    """
    rng = np.random.default_rng(seed)

    # Allow either sample_hyperellipsoid_boundary(n, 10, 4, 2)
    # or sample_hyperellipsoid_boundary(n, [10, 4, 2])
    if len(stretch) == 1 and np.ndim(stretch[0]) != 0:
        a = np.asarray(stretch[0], dtype=float)
    else:
        a = np.asarray(stretch, dtype=float)

    d = a.size

    if center is None:
        c = np.zeros(d, dtype=float)
    else:
        c = np.asarray(center, dtype=float)
        if c.shape != (d,):
            raise ValueError(f"center must have shape ({d},)")

    X = np.empty((n_points, d), dtype=float)
    filled = 0

    # Maximum possible value of sqrt(sum((u_i / a_i)^2)) over ||u||=1
    max_weight = 1.0 / np.min(a)

    while filled < n_points:
        m = min(batch_size, max(batch_size // 4, 4 * (n_points - filled)))

        # Uniform point on the unit sphere in R^d
        g = rng.normal(size=(m, d))
        u = g / np.linalg.norm(g, axis=1, keepdims=True)

        # Rejection correction to make the final ellipsoid sample surface-uniform
        weights = np.sqrt(np.sum((u / a) ** 2, axis=1))
        keep = rng.random(m) < (weights / max_weight)

        # Stretch from sphere to hyperellipsoid
        accepted = c + u[keep] * a

        take = min(len(accepted), n_points - filled)
        X[filled:filled + take] = accepted[:take]
        filled += take

    return X

def unit_ball_volume(d: int) -> float:
    return np.pi ** (d / 2) / gamma(d / 2 + 1)

def scale_axes_target_volume(stretch, target_volume: float) -> np.ndarray:
    stretch = np.asarray(stretch, dtype = float)

    d = stretch.size
    c_d = unit_ball_volume(d)

    scale = (target_volume / (c_d * np.prod(stretch))) ** (1.0 / d)
    return scale * stretch

def edge_count_curve(X: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.array([count_edges(X, t = float(t)) for t in t_grid], dtype = float)

def edge_count_curve_kde(X: np.ndarray, t_grid: np.ndarray, h: float) -> np.ndarray:
    return np.array([count_edges_kde(X, t=float(t), h=h) for t in t_grid], dtype=float)


def simulate_edge_curves_kde(point_cloud_sampler, n_resamples: int, t_grid: np.ndarray, h: float, seed: int | None):
    rng = np.random.default_rng(seed)

    curves = np.empty((n_resamples, len(t_grid)), dtype = float)

    for i in range(n_resamples):
        sample_seed = int(rng.integers(0,1_000_000_000))
        X = point_cloud_sampler(sample_seed)
        curves[i] = edge_count_curve_kde(X, t_grid, h)

    return curves

def simulate_edge_curves_kde(point_cloud_sampler, n_resamples: int, t_grid: np.ndarray, h: float, seed: int | None):
    rng = np.random.default_rng(seed)

    curves = np.empty((n_resamples, len(t_grid)), dtype = float)

    for i in range(n_resamples):
        sample_seed = int(rng.integers(0,1_000_000_000))
        X = point_cloud_sampler(sample_seed)
        curves[i] = edge_count_curve_kde(X, t_grid, h)

    return curves

def simulate_edge_curves(point_cloud_sampler, n_resamples: int, t_grid: np.ndarray, seed: int | None):
    rng = np.random.default_rng(seed)

    curves = np.empty((n_resamples, len(t_grid)), dtype = float)

    for i in range(n_resamples):
        sample_seed = int(rng.integers(0,1_000_000_000))
        X = point_cloud_sampler(sample_seed)
        curves[i] = edge_count_curve(X, t_grid)

    return curves



def plot_sphere_ellipsoid_results(result):
    t_grid = result["t_grid"]
    curves_sphere = result["curves_sphere"]
    curves_ellipsoid = result["curves_ellipsoid"]

    mean1 = curves_sphere.mean(axis=0)
    sd1 = curves_sphere.std(axis = 0, ddof = 1)

    mean2 = curves_ellipsoid.mean(axis = 0)
    sd2 = curves_ellipsoid.std(axis =0, ddof = 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(t_grid, mean1, lw = 2, label = "Sphere Edge Count Simulation Mean")
    ax[0].fill_between(t_grid, mean1 - sd1, mean1 + sd1, alpha = 0.2)

    ax[0].plot(t_grid, mean2, lw = 2, label = "Ellipsoid Edge Count Simulation Mean")
    ax[0].fill_between(t_grid, mean2 - sd2, mean2 + sd2, alpha = 0.2)

    ax[0].set_xlabel("t")
    ax[0].set_ylabel("Edge Count")
    ax[0].set_title("Edge Count Curves")
    ax[0].legend()

    ConfusionMatrixDisplay.from_predictions(
        result["y_test"],
        result["y_pred"],
        display_labels=["Sphere", "Ellipsoid"],
        cmap = "Blues",
        ax=ax[1],
    )
    ax[1].set_title("Classifier Confusion Matrix")

    plt.tight_layout()
    plt.show()

    print("Accuracy:", result["accuracy"])
    print(result["report"])


def count_triangles(X: np.ndarray, t: float) -> int:
    X = np.asarray(X, dtype=float)

    n, d = X.shape

    epsilon = t / (n ** (1 / (d - 1)))

    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) # distance matrix

    A = (D <= epsilon).astype(int) # creates adjacency matrix A where A[i,j] = 1 if points i and j are within epsilon distance otherwise A[i,j] = 0
    np.fill_diagonal(A, 0) # sets diagonal entries to 0

    A3 = A @ A @ A # entries of A^3 count the number of walks of length 3 between vertices
        
        # adds up diagonal entries of A^3 which counts all length-3 closed walks
    triangles = np.trace(A3) // 6 # divide by 6 since each triangle is counted 6 times (3 choices of starting vertex and 2 directions around the triangle)

    return int(triangles)

def count_triangles_kde(X: np.ndarray, t: float, h: float) -> int:
    X = np.asarray(X, dtype=float)

    n, d = X.shape

    epsilon = t / (n ** (1 / (d - 1)))

    D, _ = vr.weighted_distance_matrix_kde(X, h = h) # distance matrix

    A = (D <= epsilon).astype(int) # creates adjacency matrix A where A[i,j] = 1 if points i and j are within epsilon distance otherwise A[i,j] = 0
    np.fill_diagonal(A, 0) # sets diagonal entries to 0

    A3 = A @ A @ A # entries of A^3 count the number of walks of length 3 between vertices
        
        # adds up diagonal entries of A^3 which counts all length-3 closed walks
    triangles = np.trace(A3) // 6 # divide by 6 since each triangle is counted 6 times (3 choices of starting vertex and 2 directions around the triangle)

    return int(triangles)


def triangle_count_curve(X: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.array([count_triangles(X, float(t)) for t in t_grid], dtype=float)

def triangle_count_curve_kde(X: np.ndarray, t_grid: np.ndarray, h: float) -> np.ndarray:
    return np.array([count_triangles_kde(X, float(t), h = h) for t in t_grid], dtype=float)

def simulate_triangle_curves(point_cloud_sampler, n_resamples, t_grid, seed):
    rng = np.random.default_rng(seed)
    curves = np.empty((n_resamples, len(t_grid)), dtype=float)

    for i in range(n_resamples):
        sample_seed = int(rng.integers(0, 1_000_000_000))
        X = point_cloud_sampler(sample_seed)
        curves[i] = triangle_count_curve(X, t_grid)

    return curves

def simulate_triangle_curves_kde(point_cloud_sampler, n_resamples, t_grid, h, seed):
    rng = np.random.default_rng(seed)
    curves = np.empty((n_resamples, len(t_grid)), dtype=float)

    for i in range(n_resamples):
        sample_seed = int(rng.integers(0, 1_000_000_000))
        X = point_cloud_sampler(sample_seed)
        curves[i] = triangle_count_curve_kde(X, t_grid, h)

    return curves

def non_unif_hyperellipsoid(n_points, *stretch, r=1.0, center=None, seed=None):
    """
    Sample points on a hyperellipsoid boundary by sampling uniformly on the
    sphere and then stretching coordinate-wise.
    """
    rng = np.random.default_rng(seed)

    if len(stretch) == 1 and np.ndim(stretch[0]) != 0:
        a = np.asarray(stretch[0], dtype=float)
    else:
        a = np.asarray(stretch, dtype=float)

    if a.ndim != 1 or a.size == 0:
        raise ValueError("Provide one positive stretch value per dimension.")
    if np.any(a <= 0):
        raise ValueError("All stretch values must be > 0.")
    if r <= 0:
        raise ValueError("r must be > 0.")

    d = a.size

    if center is None:
        c = np.zeros(d, dtype=float)
    else:
        c = np.asarray(center, dtype=float)
        if c.shape != (d,):
            raise ValueError(f"center must have shape ({d},)")

    g = rng.normal(size=(n_points, d))
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    X = c + (r * u) * a
    return X

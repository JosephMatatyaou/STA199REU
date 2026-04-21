import numpy as np
import ecc_vr as vr
import math

def hemisphere_biased_hypersphere(
    n_points,
    d,
    p=0.8,
    r=1.0,
    center=None,
    direction=None,
    seed=None,
    shuffle=True,
):
    # p = fraction of points in the hemisphere
    # {x : <x - center, direction> >= 0}

    if d < 2:
        raise ValueError("d must be at least 2")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")

    rng = np.random.default_rng(seed)

    if direction is None:
        direction = np.zeros(d)
        direction[0] = 1.0

    direction = np.asarray(direction, dtype=float)
    if direction.shape != (d,):
        raise ValueError(f"direction must have shape ({d},)")

    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("direction must be non-zero")
    direction = direction / direction_norm

    def sample_uniform_sphere(m):
        X = rng.normal(size=(m, d))
        norms = np.linalg.norm(X, axis=1, keepdims=True)

        while np.any(norms == 0):
            bad = norms[:, 0] == 0
            X[bad] = rng.normal(size=(bad.sum(), d))
            norms[bad] = np.linalg.norm(X[bad], axis=1, keepdims=True)

        return X / norms

    n_side = int(round(p * n_points))
    n_other = n_points - n_side

    # sample points for the chosen hemisphere
    X_side = sample_uniform_sphere(n_side)
    dots_side = X_side @ direction
    X_side[dots_side < 0] *= -1.0

    # sample points for the opposite hemisphere
    X_other = sample_uniform_sphere(n_other)
    dots_other = X_other @ direction
    X_other[dots_other > 0] *= -1.0

    X = np.vstack([X_side, X_other])

    if shuffle:
        rng.shuffle(X, axis=0)

    X *= r

    if center is not None:
        center = np.asarray(center, dtype=float)
        if center.shape != (d,):
            raise ValueError(f"center must have shape ({d},)")
        X += center

    return X


def hypersphere_surface_area(d, r=1.0):
    if d < 2:
        raise ValueError("d must be at least 2")
    if r <= 0:
        raise ValueError("r must be positive")

    log_area = (
        math.log(2.0)
        + 0.5 * d * math.log(math.pi)
        + (d - 1) * math.log(r)
        - math.lgamma(d / 2.0)
    )
    return math.exp(log_area)


def hemisphere_biased_hypersphere_pdf(
    C,
    p = 0.8,
    r = 1.0,
    center = None,
    direction = None,
    tol = 1e-8,
):
    C = np.asarray(C, dtype=float)

    if C.ndim != 2:
        raise ValueError("C must have shape (n_points, d)")

    n_points, d = C.shape

    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be between 0 and 1")

    if center is None:
        center = np.zeros(d)
    center = np.asarray(center, dtype=float)
    if center.shape != (d,):
        raise ValueError(f"center must have shape ({d},)")

    if direction is None:
        direction = np.zeros(d)
        direction[0] = 1.0
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (d,):
        raise ValueError(f"direction must have shape ({d},)")

    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("direction must be non-zero")
    direction = direction / direction_norm

    shifted = C - center
    radii = np.linalg.norm(shifted, axis=1)
    on_sphere = np.abs(radii - r) <= tol

    dots = shifted @ direction
    area = hypersphere_surface_area(d, r)

    pdf_vals = np.zeros(n_points, dtype=float)
    pdf_vals[on_sphere & (dots >= 0)] = 2.0 * p / area
    pdf_vals[on_sphere & (dots < 0)] = 2.0 * (1.0 - p) / area

    return pdf_vals



def weighted_distance_matrix_pdf(
    X,
    p=0.8,
    r=1.0,
    center=None,
    direction=None,
    d_manifold=1,
    tol=1e-8,
):
  
    X = np.asarray(X, dtype=float)

    D = vr.pairwise_dist(X)

    f = hemisphere_biased_hypersphere_pdf(
        X,
        p=p,
        r=r,
        center=center,
        direction=direction,
        tol=tol,
    )

    f = np.maximum(f, 1e-12)   # guard against zero density
    f = f / f.mean()
    s = f ** (1.0 / d_manifold)   # (n,)

    si = s[:, None]
    sj = s[None, :]

    scale = (si * sj) / (si + sj)

    D_w = D * scale
    np.fill_diagonal(D_w, 0.0)
    return D_w, scale



# it should be norm less than t of n to 1 over d time ( 1 over kde(xi)^1/d)+ 1

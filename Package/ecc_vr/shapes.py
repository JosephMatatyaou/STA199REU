import numpy as np
from .metrics import random_orthonormal_matrix

def pca_project(X: np.ndarray, out_dim: int = 2) -> np.ndarray: #projects data into 2D using PCA via SVD
    Xc = X - X.mean(axis=0, keepdims=True) #centers coords to have mean 0
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False) #SVD of centered data
    W = Vt[:out_dim].T #takes out first out_dim principal directions and transpose into projection matrix
    return Xc @ W #multiply centered data with projection matrix to get lower-dimensional coordinates

def plot_point_cloud_on_ax(ax, X: np.ndarray, dim: int):
    import matplotlib.pyplot as plt
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
    
def apply_noise(Y: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    #add noise in ambient dim coords
    if noise <=0:
        return Y
    return Y + rng.normal(0, noise, size=Y.shape)

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

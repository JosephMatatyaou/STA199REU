# Make sure you installed: pip install numpy matplotlib gudhi

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

# Core Functions

def normal_point_cloud(n_points: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n_points, dim))


def build_vr_simplex_tree(X: np.ndarray, max_edge_length: float, max_simplex_dim: int) -> gd.SimplexTree:
    rips = gd.RipsComplex(points=X, max_edge_length=max_edge_length) #contructs VR complex builder
    st = rips.create_simplex_tree(max_dimension=max_simplex_dim) #builds simplex tree with simplices up do max_simplex_dim
    st.initialize_filtration() 
    return st


def compute_persistence_counts(st: gd.SimplexTree) -> dict: #takes simplex tree and returns dictionary of intervals in each homology dimension
    st.compute_persistence()
    counts = {}
    for d in range(st.dimension() + 1): #loops over dimensions in the simplex tree
        counts[d] = len(st.persistence_intervals_in_dimension(d)) #for each homology dimension in d get the list of intervals
    return counts


def compute_ecc(st: gd.SimplexTree, n_steps: int = 250): # computes ECC over the filtration
    simplices = []
    fmin, fmax = float("inf"), float("-inf")

    for simplex, filt in st.get_filtration(): #iterate through all simplices (simplex is a tuple of vertex IDs and filt is filtration value when that simplex appears)
        dim = len(simplex) - 1 #n vertices is dim n-1
        simplices.append((filt, dim)) #records when simplex appears and its dim
        fmin = min(fmin, filt) 
        fmax = max(fmax, filt)

    simplices.sort(key=lambda x: x[0]) #sort by filtration value to sweep thresholds from small to large

    ts = np.linspace(fmin, fmax, n_steps) #creates n_steps evenly spaces thresholds between min and max filtration
    max_dim = st.dimension()
    counts = np.zeros(max_dim + 1, dtype=int)

    ecc = []
    idx = 0
    for t in ts:
        while idx < len(simplices) and simplices[idx][0] <= t: #add all simplices whose filtration value <= t
            _, d = simplices[idx]
            counts[d] += 1
            idx += 1
        chi = sum(((-1) ** k) * counts[k] for k in range(max_dim + 1)) #EC formula
        ecc.append(chi)

    return ts, np.array(ecc)

# Plot helpers (one window, 3 panels)

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
        ax.set_title("Point cloud (1D)")
        return

    if dim == 2:
        ax.scatter(X[:, 0], X[:, 1], s=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Point cloud (2D)")
        ax.axis("equal")
        return

    if dim == 3:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_title("Point cloud (3D)")
        return

    # dim >= 4: show PCA 2D projection 
    X2 = pca_project(X, out_dim=2)
    ax.scatter(X2[:, 0], X2[:, 1], s=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Point cloud (PCA 2D from dim={dim})")
    ax.axis("equal")


def plot_persistence_diagram(ax, st: gd.SimplexTree):
    """
    Manual persistence diagram plot so it can be drawn on a specific matplotlib axis.
    Infinite deaths are plotted at the top boundary with triangle markers.
    """
    pers = st.persistence()  # list of (homology_dim, (birth, death))

    finite_births = []
    finite_deaths = []
    by_dim = {}

    for d, (b, de) in pers: #iterates over ever persistence interval (d:homology dim, b: birth time, de: death time)
        if d not in by_dim: #prevents mixing H0 points with H1 points
            by_dim[d] = {"b": [], "d": [], "inf_b": []}
        if np.isinf(de): #infinite death features show connected components that never merge (H0) and loops that never fill
            by_dim[d]["inf_b"].append(b)
        else:
            by_dim[d]["b"].append(b)
            by_dim[d]["d"].append(de)
            finite_births.append(b)
            finite_deaths.append(de)

    if len(finite_births) == 0: #only happens if very small VR radius
        ax.text(0.5, 0.5, "No finite intervals to plot", ha="center", va="center")
        ax.set_title("Persistence diagram")
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        return

    min_bd = min(min(finite_births), min(finite_deaths)) #plot bounds
    max_bd = max(max(finite_births), max(finite_deaths))
    pad = 0.2 * (max_bd - min_bd + 1e-9) #adds plot padding so points aren't on the border
    lo, hi = min_bd - pad, max_bd + pad #lower and upper plot limits

    # Diagonal
    ax.plot([lo, hi], [lo, hi]) #null-feature line (points near it represent short-lived features)

    # Finite points by dimension
    for d, parts in sorted(by_dim.items()): #iterates dimensions in increasing order
        if len(parts["b"]) > 0:
            ax.scatter(parts["b"], parts["d"], s=14, label=f"H{d}")

    # Infinite-death intervals: plot at y = hi with triangles
    for d, parts in sorted(by_dim.items()): 
        if len(parts["inf_b"]) > 0:
            ax.scatter(parts["inf_b"], [hi] * len(parts["inf_b"]), s=18, marker="^", label=f"H{d} (∞)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram")
    ax.legend(loc="best", fontsize=8)

def random_orthonormal_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) #random matrix
    Q, _ = np.linalg.qr(A) #Q is orthonormal matrix (random rotation matrix)
    return Q

def embed_in_ambient(X: np.ndarray, ambient_dim: int, seed: int, rotate: bool = True) -> np.ndarray:
    """
    Embed low-dim coordinates into R^ambient_dim by padding zeros, then optionally rotate.
    """
    n, d0 = X.shape #n is number of points, d0 is intrinsic dimension of shape
    if ambient_dim < d0:
        raise ValueError(f"Ambient dimension {ambient_dim} must be >= intrinsic embedding dim {d0}.")
    if ambient_dim == d0: #no embedding needed
        Y = X.copy()
    else: #pads with zeros to embed into higher-dimensional space
        Y = np.zeros((n, ambient_dim))
        Y[:, :d0] = X

    if rotate: #rotation using random orthonormal matrix so shape isn't aligned with first coords
        Q = random_orthonormal_matrix(ambient_dim, seed=seed + 12345)
        Y = Y @ Q
    return Y

def sample_point_cloud( #shape generator
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
    rotate: bool = True,
) -> np.ndarray:
    """
    Supported shapes:
      - "Normal Blob" (normal point cloud) in R^ambient_dim
      - "Circle"   (S^1) embedded in R^2 then into R^ambient_dim
      - "Cylinder" (S^1 x [0,1]) embedded in R^3 then into R^ambient_dim
      - "Torus"    (S^1 x S^1) embedded in R^3 then into R^ambient_dim
    """
    rng = np.random.default_rng(seed)

    shape = shape.lower().strip()

    if shape == "normal blob":
        X = rng.normal(0.0, 1.0, size=(n_points, ambient_dim)) #standard gaussian cloud in R^d
        if noise > 0: #adds additional noise
            X = X + rng.normal(0.0, noise, size=X.shape)
        return X

    if shape == "circle":
        if ambient_dim < 2:
            raise ValueError("Circle needs ambient_dim >= 2.")
        theta = rng.uniform(0, 2 * np.pi, size=n_points) #uniform sampling on S^1
        # Gaussian thickness around the circle: radius = R + N(0, noise)
        r = circle_radius + (rng.normal(0.0, noise, size=n_points) if noise > 0 else 0.0)
        # Ensure radius stays positive
        r = np.maximum(r, 1e-6)
        X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])  #polar to cartesian conversion
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate)

    if shape == "cylinder":
        if ambient_dim < 3:
            raise ValueError("Cylinder needs ambient_dim >= 3.")
        theta = rng.uniform(0, 2 * np.pi, size=n_points) #angular coord
        # Gaussian thickness in radius
        r = cylinder_radius + (rng.normal(0.0, noise, size=n_points) if noise > 0 else 0.0)
        r = np.maximum(r, 1e-6)
        # Gaussian distribution along the axis, truncated to the cylinder height
        if cylinder_height <= 0:
            raise ValueError("cylinder_height must be > 0.")
        z = rng.normal(0.0, cylinder_height / 4.0, size=n_points)
        z = np.clip(z, -cylinder_height / 2.0, cylinder_height / 2.0)
        X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])  # R^3
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate)

    if shape == "torus":
        if ambient_dim < 3:
            raise ValueError("Torus needs ambient_dim >= 3.")
        theta = rng.uniform(0, 2 * np.pi, size=n_points)
        phi = rng.uniform(0, 2 * np.pi, size=n_points)
        # Gaussian thickness around the torus tube: minor radius = r + N(0, noise)
        rr = Torus_r + (rng.normal(0.0, noise, size=n_points) if noise > 0 else 0.0)
        rr = np.maximum(rr, 1e-6)
        x = (Torus_R + rr * np.cos(phi)) * np.cos(theta)
        y = (Torus_R + rr * np.cos(phi)) * np.sin(theta)
        z = rr * np.sin(phi)
        X0 = np.column_stack([x, y, z])  # R^3
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate)

    raise ValueError(f"Unknown shape: {shape}. Choose normal blob, circle, cylinder, torus.")


# GUI 

class ECCApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VR Filtration → Point Cloud + Persistence Diagram + ECC")
        self.geometry("600x520")
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        title = ttk.Label(self, text="Normal Point Cloud → VR → PD + ECC", font=("Helvetica", 14, "bold"))
        title.pack(pady=10)

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        self.dim_var = tk.StringVar(value="3")
        self.npoints_var = tk.StringVar(value="200")
        self.maxradius_var = tk.StringVar(value="2.0")
        self.maxsimp_var = tk.StringVar(value="3")
        self.seed_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="250")
        self.shape_var = tk.StringVar(value="Normal Blob")
        shape_options = ["Normal Blob", "Circle", "Cylinder", "Torus"]
        self.noise_var = tk.StringVar(value= "0.05")

        self._row(frm, "Dimension:", self.dim_var, 0)
        self._row(frm, "Number of Points:", self.npoints_var, 1)
        self._row(frm, "Max Radius length:", self.maxradius_var, 2)
        self._row(frm, "Max Simplex Dimension:", self.maxsimp_var, 3)
        self._row(frm, "Random Seed:", self.seed_var, 4)
        self._row(frm, "ECC Steps (Resolution):", self.steps_var, 5)
        self._row(frm, "Noise Standard Deviation:", self.noise_var, 6)

        ttk.Label(frm, text="Shape:").grid(row=7, column=0, sticky="w", padx=6, pady=4)
        shape_box = ttk.Combobox(
            frm,
            textvariable=self.shape_var,
            values=shape_options,
            state="readonly",
            width=18
        )
        shape_box.grid(row=7, column=1, sticky="w", padx=6, pady=4)


        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Run (show all 3 plots)", command=self.on_run).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        outfrm = ttk.LabelFrame(self, text="Log")
        outfrm.pack(fill="both", expand=True, **pad)

        self.output = tk.Text(outfrm, height=8, wrap="word")
        self.output.pack(fill="both", expand=True, padx=8, pady=8)

        self._log("Tip: If it’s slow/error, reduce max_simplex_dim (try 2), then max_radius_length, then n_points.\n")

    def _row(self, parent, label, var, r):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=var, width=20).grid(row=r, column=1, sticky="w", padx=6, pady=4)

    def _log(self, s: str):
        self.output.insert("end", s + "\n")
        self.output.see("end")

    def on_run(self):
        
        # Parse + validate
        try:
            dim = int(self.dim_var.get())
            n_points = int(self.npoints_var.get())
            max_radius = float(self.maxradius_var.get())
            max_simp = int(self.maxsimp_var.get())
            seed = int(self.seed_var.get())
            steps = int(self.steps_var.get())
            shape = self.shape_var.get()
            noise = float(self.noise_var.get())

            shape_label = "Normal Blob" if shape == "Normal Blob" else shape.capitalize()

            if dim < 1:
                raise ValueError("Dimension must be >= 1.")
            if n_points < 2:
                raise ValueError("n_points must be >= 2.")
            if max_radius <= 0:
                raise ValueError("max_radius_length must be > 0.")
            if max_simp < 1:
                raise ValueError("max_simplex_dim 
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
    rips = gd.RipsComplex(points=X, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_simplex_dim)
    st.initialize_filtration()
    return st


def compute_persistence_counts(st: gd.SimplexTree) -> dict:
    st.compute_persistence()
    counts = {}
    for d in range(st.dimension() + 1):
        counts[d] = len(st.persistence_intervals_in_dimension(d))
    return counts


def compute_ecc(st: gd.SimplexTree, n_steps: int = 250):
    simplices = []
    fmin, fmax = float("inf"), float("-inf")

    for simplex, filt in st.get_filtration():
        dim = len(simplex) - 1
        simplices.append((filt, dim))
        fmin = min(fmin, filt)
        fmax = max(fmax, filt)

    simplices.sort(key=lambda x: x[0])

    ts = np.linspace(fmin, fmax, n_steps)
    max_dim = st.dimension()
    counts = np.zeros(max_dim + 1, dtype=int)

    ecc = []
    idx = 0
    for t in ts:
        while idx < len(simplices) and simplices[idx][0] <= t:
            _, d = simplices[idx]
            counts[d] += 1
            idx += 1
        chi = sum(((-1) ** k) * counts[k] for k in range(max_dim + 1))
        ecc.append(chi)

    return ts, np.array(ecc)

# Plot helpers (one window, 3 panels)

def pca_project(X: np.ndarray, out_dim: int = 2) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:out_dim].T
    return Xc @ W


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


def plot_persistence_diagram_on_ax(ax, st: gd.SimplexTree):
    """
    Manual persistence diagram plot so it can be drawn on a specific matplotlib axis.
    Infinite deaths are plotted at the top boundary with triangle markers.
    """
    pers = st.persistence()  # list of (homology_dim, (birth, death))

    finite_births = []
    finite_deaths = []
    by_dim = {}

    for d, (b, de) in pers:
        if d not in by_dim:
            by_dim[d] = {"b": [], "d": [], "inf_b": []}
        if np.isinf(de):
            by_dim[d]["inf_b"].append(b)
        else:
            by_dim[d]["b"].append(b)
            by_dim[d]["d"].append(de)
            finite_births.append(b)
            finite_deaths.append(de)

    if len(finite_births) == 0:
        ax.text(0.5, 0.5, "No finite intervals to plot", ha="center", va="center")
        ax.set_title("Persistence diagram")
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        return

    min_bd = min(min(finite_births), min(finite_deaths))
    max_bd = max(max(finite_births), max(finite_deaths))
    pad = 0.05 * (max_bd - min_bd + 1e-9)
    lo, hi = min_bd - pad, max_bd + pad

    # Diagonal
    ax.plot([lo, hi], [lo, hi])

    # Finite points by dimension
    for d, parts in sorted(by_dim.items()):
        if len(parts["b"]) > 0:
            ax.scatter(parts["b"], parts["d"], s=14, label=f"H{d}")

    # Infinite-death intervals: plot at y = hi with triangles
    for d, parts in sorted(by_dim.items()):
        if len(parts["inf_b"]) > 0:
            ax.scatter(parts["inf_b"], [hi] * len(parts["inf_b"]), s=18, marker="^", label=f"H{d} (∞)")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("Persistence diagram")
    ax.legend(loc="best", fontsize=8)

def random_orthonormal_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    return Q

def embed_in_ambient(X: np.ndarray, ambient_dim: int, seed: int, rotate: bool = True) -> np.ndarray:
    """
    Embed low-dim coordinates into R^ambient_dim by padding zeros, then optionally rotate.
    """
    n, d0 = X.shape
    if ambient_dim < d0:
        raise ValueError(f"Ambient dimension {ambient_dim} must be >= intrinsic embedding dim {d0}.")
    if ambient_dim == d0:
        Y = X.copy()
    else:
        Y = np.zeros((n, ambient_dim))
        Y[:, :d0] = X

    if rotate:
        Q = random_orthonormal_matrix(ambient_dim, seed=seed + 12345)
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
    torus_R: float = 2.0,
    torus_r: float = 0.7,
    rotate: bool = True,
) -> np.ndarray:
    """
    Supported shapes:
      - "gaussian" (normal point cloud) in R^ambient_dim
      - "circle"   (S^1) embedded in R^2 then into R^ambient_dim
      - "cylinder" (S^1 x [0,1]) embedded in R^3 then into R^ambient_dim
      - "torus"    (S^1 x S^1) embedded in R^3 then into R^ambient_dim
    """
    rng = np.random.default_rng(seed)

    shape = shape.lower().strip()

    if shape == "gaussian":
        X = rng.normal(0.0, 1.0, size=(n_points, ambient_dim))
        if noise > 0:
            X = X + rng.normal(0.0, noise, size=X.shape)
        return X

    if shape == "circle":
        if ambient_dim < 2:
            raise ValueError("Circle needs ambient_dim >= 2.")
        theta = rng.uniform(0, 2 * np.pi, size=n_points)
        X0 = np.column_stack([circle_radius * np.cos(theta),
                              circle_radius * np.sin(theta)])  # R^2
        if noise > 0:
            X0 = X0 + rng.normal(0.0, noise, size=X0.shape)
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate)

    if shape == "cylinder":
        if ambient_dim < 3:
            raise ValueError("Cylinder needs ambient_dim >= 3.")
        theta = rng.uniform(0, 2 * np.pi, size=n_points)
        z = rng.uniform(-cylinder_height / 2, cylinder_height / 2, size=n_points)
        X0 = np.column_stack([cylinder_radius * np.cos(theta),
                              cylinder_radius * np.sin(theta),
                              z])  # R^3
        if noise > 0:
            X0 = X0 + rng.normal(0.0, noise, size=X0.shape)
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate)

    if shape == "torus":
        if ambient_dim < 3:
            raise ValueError("Torus needs ambient_dim >= 3.")
        theta = rng.uniform(0, 2 * np.pi, size=n_points)
        phi = rng.uniform(0, 2 * np.pi, size=n_points)
        x = (torus_R + torus_r * np.cos(phi)) * np.cos(theta)
        y = (torus_R + torus_r * np.cos(phi)) * np.sin(theta)
        z = torus_r * np.sin(phi)
        X0 = np.column_stack([x, y, z])  # R^3
        if noise > 0:
            X0 = X0 + rng.normal(0.0, noise, size=X0.shape)
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate)

    raise ValueError(f"Unknown shape: {shape}. Choose gaussian, circle, cylinder, torus.")


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
        self.maxedge_var = tk.StringVar(value="2.0")
        self.maxsimp_var = tk.StringVar(value="3")
        self.seed_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="250")
        self.shape_var = tk.StringVar(value = "Gaussian")
        shape_options = ["Gaussian", "Circle", "Cylinder", "Torus"]
        self.noise_var = tk.StringVar(value= "0.05")

        self._row(frm, "Dimension (e.g., 3 or 4):", self.dim_var, 0)
        self._row(frm, "# Points (n_points):", self.npoints_var, 1)
        self._row(frm, "Max edge length:", self.maxedge_var, 2)
        self._row(frm, "Max simplex dimension:", self.maxsimp_var, 3)
        self._row(frm, "Random seed:", self.seed_var, 4)
        self._row(frm, "ECC steps (resolution):", self.steps_var, 5)
        self._row(frm, "Noise Std:", self.noise_var, 6)

        ttk.Label(frm, text="Shape:").grid(row=7, column = 0, sticky = "w", padx=6, pady=4)
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

        self._log("Tip: If it’s slow/error, reduce max_simplex_dim (try 2), then max_edge_length, then n_points.\n")

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
            max_edge = float(self.maxedge_var.get())
            max_simp = int(self.maxsimp_var.get())
            seed = int(self.seed_var.get())
            steps = int(self.steps_var.get())
            shape = self.shape_var.get()
            noise = float(self.noise_var.get())

            if dim < 1:
                raise ValueError("Dimension must be >= 1.")
            if n_points < 2:
                raise ValueError("n_points must be >= 2.")
            if max_edge <= 0:
                raise ValueError("max_edge_length must be > 0.")
            if max_simp < 1:
                raise ValueError("max_simplex_dim must be >= 1.")
            if steps < 10:
                raise ValueError("ECC steps must be >= 10.")
            if noise < 0:
                raise ValueError("Noise must be >= 0.")
        except Exception as e:
            messagebox.showerror("Invalid input", f"Please check your inputs.\n\nDetails: {e}")
            return

        self._log("----- Running -----")
        self._log(f"dim={dim}, n_points={n_points}, max_edge_length={max_edge}, max_simplex_dim={max_simp}, seed={seed}")

        try:
            # 1) Data
            X = sample_point_cloud(
                shape=shape,
                n_points=n_points,
                ambient_dim=dim,
                seed=seed,
                noise=noise,
                circle_radius=1,
                cylinder_radius=1,
                cylinder_height=2,
                torus_R=2,
                torus_r=0.7,
                rotate=True,
            )

            # 2) VR filtration
            st = build_vr_simplex_tree(X, max_edge, max_simp)
            self._log(f"Total simplices: {st.num_simplices()}")

            # 3) Persistence
            pers_counts = compute_persistence_counts(st)
            for d in sorted(pers_counts):
                self._log(f"H{d} intervals: {pers_counts[d]}")

            # 4) ECC
            t, ecc = compute_ecc(st, n_steps=steps)

            # 5) ONE figure: point cloud + persistence diagram + ECC
            fig = plt.figure(figsize=(14, 4))

            # Panel 1: point cloud
            if dim == 3:
                ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            else:
                ax1 = fig.add_subplot(1, 3, 1)
            plot_point_cloud_on_ax(ax1, X, dim)

            # Panel 2: persistence diagram
            ax2 = fig.add_subplot(1, 3, 2)
            plot_persistence_diagram_on_ax(ax2, st)

            # Panel 3: ECC
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(t, ecc)
            ax3.set_xlabel("filtration threshold")
            ax3.set_ylabel("Euler characteristic χ(t)")
            ax3.set_title("ECC")

            fig.suptitle(f"VR results (shape={shape}, dim={dim}, n={n_points}, max_edge={max_edge}, max_simp={max_simp})")
            fig.tight_layout()
            plt.show()

            self._log("Done. (All 3 plots shown in one window.)")

        except Exception as e:
            messagebox.showerror(
                "Computation error",
                "Something went wrong (often VR got too large).\n"
                "Try smaller n_points, smaller max_edge_length, or smaller max_simplex_dim.\n\n"
                f"Details: {e}"
            )
            self._log(f"ERROR: {e}")


if __name__ == "__main__":
    app = ECCApp()
    app.mainloop()
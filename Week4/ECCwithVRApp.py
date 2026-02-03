# Make sure you installed: pip install numpy matplotlib gudhi sv_ttk
# Joseph Matatyaou

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import sv_ttk

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
        X = rng.normal(0.0, 1.0, size=(n_points, ambient_dim)) #standard gaussian cloud in R^d
        if noise > 0: #adds additional noise
            X = X + rng.normal(0.0, noise, size=X.shape)
        return X

    if shape_key == "circle":
        if ambient_dim < 2:
            raise ValueError("Circle needs ambient_dim >= 2.")
        theta = rng.uniform(0, 2 * np.pi, size=n_points) #uniform sampling on S^1
        # Gaussian thickness around the circle: radius = R + N(0, noise)
        r = circle_radius + (rng.normal(0.0, noise, size=n_points) if noise > 0 else 0.0)
        # Ensure radius stays positive
        r = np.maximum(r, 1e-6)
        X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])  #polar to cartesian conversion
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)
    
    if shape_key in {"filleddisk", "disk", "filleddisc", "disc"}:
        #filled 2D disk. For uniform area sample radius as R*sqrt(U)
        if ambient_dim < 2:
            raise ValueError("Filled Disk needs ambient_dim >= 2.")
        
        theta = rng.uniform(0, 2 * np.pi, size = n_points)
        u = rng.uniform(0, 1, size=n_points)
        r = circle_radius * np.sqrt(u)

        #noise
        if noise > 0:
            r = r + rng.normal(0, noise, size=n_points)
        r = np.maximum(r, 0)

        X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)

    if shape_key in {"figure8", "figureeight"}:
        # figure 8 (wedge of two circles): two loops that connect at one point.
        # Two circles of radius R/2 centered at +/- (R/2, 0), so they intersect at the origin.

        if ambient_dim < 2:
            raise ValueError("Figure 8 needs ambient_dim >= 2.")
        
        n1 = n_points // 2
        n2 = n_points - n1

        theta1 = rng.uniform(0, 2 * np.pi, size = n1)
        theta2 = rng.uniform(0, 2 * np.pi, size = n2)

        Rloop = circle_radius / 2

        r1 = Rloop + (rng.normal(0, noise, size = n1) if noise > 0 else np.zeros(n1))
        r2 = Rloop + (rng.normal(0, noise, size = n2) if noise > 0 else np.zeros(n2))
        r1 = np.maximum(r1, 1e-6)
        r2 = np.maximum(r2, 1e-6)

        c1 = np.array([-circle_radius / 2, 0])
        c2 = np.array([circle_radius / 2, 0])

        X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)]) + c1
        X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)]) + c2

        X0 = np.vstack([X1, X2])
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)

    if shape_key == "cylinder":
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
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)

    if shape_key in {"closedcylinder", "cylinderwithcaps", "cylinderfilledends", "closed"}:
        if ambient_dim < 3:
            raise ValueError("Closed Cylinder needs ambient_dim >= 3.")
        if cylinder_height <= 0:
            raise ValueError("Cylinder Height must be > 0.")
        
        # Mix points between side and caps
        n_side = int(round(.6 * n_points))
        n_cap_each = (n_points - n_side) // 2
        n_top = n_cap_each
        n_bot = n_points - n_side - n_top

        # Shell
        theta_s = rng.uniform(0, 2 * np.pi, size=n_side)
        r_s = cylinder_radius + (rng.normal(0.0, noise, size=n_side) if noise > 0 else np.zeros(n_side))
        r_s = np.maximum(r_s, 1e-6)
        z_s = rng.uniform(-cylinder_height / 2, cylinder_height / 2, size=n_side)
        Xs = np.column_stack([r_s * np.cos(theta_s), r_s * np.sin(theta_s), z_s])

        # Caps
        def sample_cap(n, z0):
            # Filled disk cap in the xy-plane at height z0
            th = rng.uniform(0, 2 * np.pi, size=n)
            u = rng.uniform(0.0, 1.0, size=n)
            rr = cylinder_radius * np.sqrt(u)  # uniform in area
            if noise > 0:
                rr = rr + rng.normal(0.0, noise, size=n)
            rr = np.maximum(rr, 0.0)
            x = rr * np.cos(th)
            y = rr * np.sin(th)
            z = np.full(n, z0)
            return np.column_stack([x, y, z])
        
        Xt = sample_cap(n_top, cylinder_height / 2)
        Xb = sample_cap(n_bot, -cylinder_height / 2)

        X0 = np.vstack([Xs, Xt, Xb])
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)

    if shape_key == "sphere":
        if ambient_dim < 3:
            raise ValueError("Sphere needs ambient_dim >= 3.")

        # Sample directions uniformly on S^2
        v = rng.normal(size=(n_points, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        # Gaussian thickness around the sphere: radius = R + N(0, noise)
        rad = sphere_radius + (rng.normal(0.0, noise, size=n_points) if noise > 0 else np.zeros(n_points))
        rad = np.maximum(rad, 1e-6)

        X0 = v * rad[:, None]  # (n_points, 3)
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)

#####recode torus to uniformly distribute points (look at parametric family non uniform torus?)

    if shape_key == "torus":
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
        return embed_in_ambient(X0, ambient_dim, seed=seed, rotate=rotate, rotate_seed=rotate_seed)

    if shape_key == "swissroll":
        if ambient_dim < 3:
            raise ValueError("Swiss Roll needs ambient_dim >= 3")
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size = n_points)
        h = rng.uniform(-1, 1, size=n_points)

        # noise
        if noise > 0:
            t = t + rng.normal(0, noise, size = n_points)
            h = h + rng.normal(0, noise, size = n_points)
        
        x = t * np.cos(t)
        y = h
        z = t * np.sin(t)
        X0 = np.column_stack([x, y, z])
        
        return embed_in_ambient(X0, ambient_dim, seed = seed, rotate = rotate, rotate_seed=rotate_seed)
    
    raise ValueError(f"Unknown shape: {shape}. Choose normal blob, circle, disk, figure 8, cylinder, closed cylinder sphere, torus, or swiss roll.")

# GUI 

class ECCApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECC with Vietoris–Rips App")
        self.geometry("600x600")
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        title = ttk.Label(self, text="Point Cloud → Vietoris–Rips → Persistence Diagram + ECC", font=("Helvetica", 14, "bold"))
        title.pack(pady=10)

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        self.dim_var = tk.StringVar(value="3")
        self.npoints_var = tk.StringVar(value="100")
        self.maxradius_var = tk.StringVar(value="2.0")
        self.maxsimp_var = tk.StringVar(value="2")
        self.seed_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="250")
        self.shape_var = tk.StringVar(value="Normal Blob")
        shape_options = [
            "Normal Blob",
            "Circle",
            "Filled Disk",
            "Figure 8",
            "Sphere",
            "Cylinder",
            "Closed Cylinder",
            "Torus",
            "Swiss Roll",
        ]
        self.noise_var = tk.StringVar(value= "0.05")
        self.rotate_var = tk.BooleanVar(value=True)
        self.rotate_seed_var = tk.StringVar(value="")

        self._row(frm, "Dimension:", self.dim_var, 0)
        self._row(frm, "Number of Points:", self.npoints_var, 1)
        self._row(frm, "Max Radius length:", self.maxradius_var, 2)
        self._row(frm, "Max Simplex Dimension:", self.maxsimp_var, 3)
        self._row(frm, "Random Seed:", self.seed_var, 4)
        self._row(frm, "ECC Steps (Resolution):", self.steps_var, 5)
        self._row(frm, "Noise Standard Deviation:", self.noise_var, 6)

        # Rotation controls
        ttk.Label(frm, text="Rotate Shape:").grid(row=7, column=0, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(frm, variable=self.rotate_var).grid(row=7, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frm, text="Rotation Seed:").grid(row=8, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frm, textvariable=self.rotate_seed_var, width=20).grid(row=8, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frm, text="Shape:").grid(row=9, column=0, sticky="w", padx=6, pady=4)
        shape_box = ttk.Combobox(
            frm,
            textvariable=self.shape_var,
            values=shape_options,
            state="readonly",
            width=18
        )
        shape_box.grid(row=9, column=1, sticky="w", padx=6, pady=4)


        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        ttk.Button(btns, text="Run", command=self.on_run).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        outfrm = ttk.LabelFrame(self, text="Log")
        outfrm.pack(fill="both", expand=True, **pad)

        self.output = tk.Text(outfrm, height=8, wrap="word")
        self.output.pack(fill="both", expand=True, padx=8, pady=8)

        self._log("If it’s slow/error, reduce max simplex dimension, then max radius length, then sampling points.\n")

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
            rotate = bool(self.rotate_var.get())
            rot_seed_txt = self.rotate_seed_var.get().strip()
            rotate_seed = int(rot_seed_txt) if rot_seed_txt != "" else None

            shape_label = "Normal Blob" if shape == "Normal Blob" else shape.capitalize()

            if dim < 1:
                raise ValueError("Dimension must be >= 1.")
            if n_points < 2:
                raise ValueError("n_points must be >= 2.")
            if max_radius <= 0:
                raise ValueError("max_radius_length must be > 0.")
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
        self._log(
            f"shape={shape_label}, dim={dim}, n_points={n_points}, max_radius_length={max_radius}, "
            f"max_simplex_dim={max_simp}, seed={seed}, rotate={rotate}, rotate_seed={rotate_seed}"
        )

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
                Torus_R=2,
                Torus_r=1.9,
                rotate=rotate,
                rotate_seed=rotate_seed,
            )

            # 2) VR filtration
            st = build_vr_simplex_tree(X, max_radius, max_simp)
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
            plot_persistence_diagram(ax2, st)

            # Panel 3: ECC
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(t, ecc)
            ax3.set_xlabel("VR Ball Radius")
            ax3.set_ylabel("Euler characteristic")
            ax3.set_title("ECC")

            fig.suptitle(f"VR results (Shape={shape_label}, Dim={dim}, n={n_points}, Max Radius ={max_radius}, Max Simplex Dim={max_simp})")
            fig.tight_layout()
            plt.show()

            self._log("Done. (All 3 plots shown in one window.)")

        except Exception as e:
            messagebox.showerror(
                "Computation error",
                "Something went wrong (often VR got too large).\n"
                "Try smaller n_points, smaller max_radius_length, or smaller max_simplex_dim.\n\n"
                f"Details: {e}"
            )
            self._log(f"ERROR: {e}")


if __name__ == "__main__":
    app = ECCApp()
    sv_ttk.set_theme("dark")
    app.mainloop()
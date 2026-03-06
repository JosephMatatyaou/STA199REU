import numpy as np
import matplotlib.pyplot as plt

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
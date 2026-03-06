"""
ECC VR - Euler Characteristic Curve and Vietoris-Rips Complex Computations

A fast library for topological data analysis including:
- Vietoris-Rips complex enumeration
- Euler characteristic computations
- Point cloud sampling and visualization
- Kernel density estimation and density-weighted filtrations
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Fast VR complex and Euler characteristic computations"


# CORE VR COMPLEX FUNCTIONS

from .core import (
    Nplus,
    order_vertices_by_eps_neighbors,
    local_contributions_vr,
    EC_from_C,
    ECC_from_C,
    plot_ECC_from_C,
    chi,
    run_ec_simulation,
)


# METRICS AND UTILITIES

from .metrics import (
    pairwise_dist,
    intersect_two_pointer,
    random_orthonormal_matrix,
    fhat,
    weighted_distance_matrix_kde,
)


# SHAPE SAMPLING AND VISUALIZATION

from .shapes import (
    pca_project,
    plot_point_cloud_on_ax,
    apply_noise,
    embed_in_ambient,
    unif_torus_points,
    sample_point_cloud,
)


# PUBLIC API

__all__ = [
    # Core VR functions
    "Nplus",
    "order_vertices_by_eps_neighbors",
    "local_contributions_vr",
    "EC_from_C",
    "ECC_from_C",
    "plot_ECC_from_C",
    "chi",
    "run_ec_simulation",
    # Metrics and utilities
    "pairwise_dist",
    "intersect_two_pointer",
    "random_orthonormal_matrix",
    "fhat",
    "weighted_distance_matrix_kde",
    # Shapes
    "pca_project",
    "plot_point_cloud_on_ax",
    "apply_noise",
    "embed_in_ambient",
    "unif_torus_points",
    "sample_point_cloud",
]
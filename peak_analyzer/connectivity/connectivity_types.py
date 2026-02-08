import numpy as np
from scipy.ndimage import generate_binary_structure

class Connectivity:
    """
    Implement n-dimensional k-connectivity with minimal code.
    k=1: face connectivity, k=ndim: full (Moore) connectivity.
    """
    def __init__(self, ndim: int, k: int):
        if not (1 <= k <= ndim):
            raise ValueError(f"k must be between 1 and {ndim}")
        
        self.ndim = ndim
        self.k = k
        self.structure = generate_binary_structure(ndim, k)

        # Set the center element (self) to False before extraction.
        # Since all dimensions are size 3, the center flat index is uniquely size // 2.
        self.structure.flat[self.structure.size // 2] = False
        self.offsets = np.argwhere(self.structure) - 1

    def get_neighbors(self, center: tuple, shape: tuple) -> np.ndarray:
        """Return valid neighboring coordinates with boundary conditions applied."""
        candidates = np.asarray(center) + self.offsets
        valid = np.all((candidates >= 0) & (candidates < shape), axis=1)
        return candidates[valid]

    @property
    def neighbor_count(self) -> int:
        return len(self.offsets)

    def __repr__(self) -> str:
        return f"Connectivity(n={self.ndim}, k={self.k}, count={self.neighbor_count})"
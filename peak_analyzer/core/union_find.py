"""
Union-Find Data Structure

Implements efficient Union-Find (Disjoint Set Union) data structure
for connected component analysis in peak detection.
"""

from typing import Any
import numpy as np


class UnionFind:
    """
    Union-Find data structure with path compression and union by rank.
    
    Efficiently maintains disjoint sets and supports union and find operations
    in nearly constant amortized time.
    """
    
    def __init__(self, size: int | None = None, elements: list[Any | None] = None):
        """
        Initialize Union-Find structure.
        
        Parameters:
        -----------
        size : int, optional
            Number of elements (for integer-indexed elements)
        elements : list, optional
            List of arbitrary elements to manage
        """
        if size is not None:
            self.parent = list(range(size))
            self.rank = [0] * size
            self.size = [1] * size
            self.num_components = size
            self._element_to_id = {i: i for i in range(size)}
            self._id_to_element = {i: i for i in range(size)}
            
        elif elements is not None:
            n = len(elements)
            self.parent = list(range(n))
            self.rank = [0] * n
            self.size = [1] * n
            self.num_components = n
            self._element_to_id = {elem: i for i, elem in enumerate(elements)}
            self._id_to_element = {i: elem for i, elem in enumerate(elements)}
            
        else:
            self.parent = []
            self.rank = []
            self.size = []
            self.num_components = 0
            self._element_to_id = {}
            self._id_to_element = {}
    
    def add_element(self, element: Any) -> int:
        """
        Add new element to the Union-Find structure.
        
        Parameters:
        -----------
        element : Any
            Element to add
            
        Returns:
        --------
        int
            ID assigned to the element
        """
        if element in self._element_to_id:
            return self._element_to_id[element]
            
        new_id = len(self.parent)
        self.parent.append(new_id)
        self.rank.append(0)
        self.size.append(1)
        self._element_to_id[element] = new_id
        self._id_to_element[new_id] = element
        self.num_components += 1
        
        return new_id
    
    def find(self, x: Any) -> int:
        """
        Find root of the set containing x with path compression.
        
        Parameters:
        -----------
        x : Any
            Element to find root for
            
        Returns:
        --------
        int
            ID of the root element
        """
        # Convert element to ID if necessary
        if isinstance(x, int) and x < len(self.parent):
            x_id = x
        else:
            if x not in self._element_to_id:
                raise ValueError(f"Element {x} not found in Union-Find structure")
            x_id = self._element_to_id[x]
        
        # Path compression
        if self.parent[x_id] != x_id:
            self.parent[x_id] = self.find(self.parent[x_id])
        
        return self.parent[x_id]
    
    def union(self, x: Any, y: Any) -> bool:
        """
        Union two sets containing x and y using union by rank.
        
        Parameters:
        -----------
        x, y : Any
            Elements to union
            
        Returns:
        --------
        bool
            True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        # Attach smaller rank tree under root of higher rank tree
        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        
        # If ranks were equal, increase rank of new root
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        self.num_components -= 1
        return True
    
    def connected(self, x: Any, y: Any) -> bool:
        """
        Check if two elements are in the same connected component.
        
        Parameters:
        -----------
        x, y : Any
            Elements to check
            
        Returns:
        --------
        bool
            True if elements are connected
        """
        try:
            return self.find(x) == self.find(y)
        except ValueError:
            return False
    
    def get_components(self) -> dict[int, list[Any]]:
        """
        Get all connected components.
        
        Returns:
        --------
        dict[int, list[Any]]
            Dictionary mapping root IDs to lists of elements in component
        """
        components = {}
        
        for element_id in range(len(self.parent)):
            root = self.find(element_id)
            element = self._id_to_element[element_id]
            
            if root not in components:
                components[root] = []
            components[root].append(element)
        
        return components
    
    def get_component_sizes(self) -> dict[int, int]:
        """
        Get sizes of all connected components.
        
        Returns:
        --------
        dict[int, int]
            Dictionary mapping root IDs to component sizes
        """
        sizes = {}
        
        for element_id in range(len(self.parent)):
            root = self.find(element_id)
            if root not in sizes:
                sizes[root] = self.size[root]
        
        return sizes
    
    def get_component_containing(self, x: Any) -> list[Any]:
        """
        Get all elements in the same component as x.
        
        Parameters:
        -----------
        x : Any
            Element to get component for
            
        Returns:
        --------
        list[Any]
            List of all elements in same component
        """
        root = self.find(x)
        components = self.get_components()
        return components.get(root, [])
    
    def component_count(self) -> int:
        """
        Get total number of connected components.
        
        Returns:
        --------
        int
            Number of connected components
        """
        return self.num_components
    
    def component_size(self, x: Any) -> int:
        """
        Get size of component containing element x.
        
        Parameters:
        -----------
        x : Any
            Element to get component size for
            
        Returns:
        --------
        int
            Size of component containing x
        """
        root = self.find(x)
        return self.size[root]
    
    def merge_components(self, component_elements: list[list[Any]]):
        """
        Merge multiple components defined by lists of elements.
        
        Parameters:
        -----------
        component_elements : list[list[Any]]
            List of lists, where each inner list contains elements to be merged
        """
        for component in component_elements:
            if len(component) > 1:
                # Union all elements in this component
                first = component[0]
                for element in component[1:]:
                    self.union(first, element)
    
    def reset(self):
        """Reset Union-Find structure to initial state."""
        for i in range(len(self.parent)):
            self.parent[i] = i
            self.rank[i] = 0
            self.size[i] = 1
        self.num_components = len(self.parent)


class GridUnionFind:
    """
    Specialized Union-Find for grid-based data structures.
    
    Optimized for working with N-dimensional grid coordinates efficiently.
    """
    
    def __init__(self, shape: tuple[int, ...]):
        """
        Initialize grid-based Union-Find.
        
        Parameters:
        -----------
        shape : tuple of int
            Shape of the N-dimensional grid
        """
        self.shape = shape
        self.ndim = len(shape)
        self.size = np.prod(shape)
        
        # Initialize Union-Find for flattened indices
        self.uf = UnionFind(int(self.size))
        
    def _coord_to_index(self, coord: tuple[int, ...]) -> int:
        """Convert N-dimensional coordinate to flat index."""
        if len(coord) != self.ndim:
            raise ValueError(f"Coordinate dimension mismatch: expected {self.ndim}, got {len(coord)}")
        
        # Check bounds
        for i, (c, s) in enumerate(zip(coord, self.shape)):
            if not 0 <= c < s:
                raise ValueError(f"Coordinate {coord} out of bounds for shape {self.shape}")
        
        # Convert to flat index using numpy's ravel_multi_index
        return int(np.ravel_multi_index(coord, self.shape))
    
    def _index_to_coord(self, index: int) -> tuple[int, ...]:
        """Convert flat index to N-dimensional coordinate."""
        return tuple(np.unravel_index(index, self.shape))
    
    def union_coords(self, coord1: tuple[int, ...], coord2: tuple[int, ...]) -> bool:
        """
        Union two grid coordinates.
        
        Parameters:
        -----------
        coord1, coord2 : tuple of int
            Grid coordinates to union
            
        Returns:
        --------
        bool
            True if union was performed
        """
        idx1 = self._coord_to_index(coord1)
        idx2 = self._coord_to_index(coord2)
        return self.uf.union(idx1, idx2)
    
    def connected_coords(self, coord1: tuple[int, ...], coord2: tuple[int, ...]) -> bool:
        """
        Check if two grid coordinates are connected.
        
        Parameters:
        -----------
        coord1, coord2 : tuple of int
            Grid coordinates to check
            
        Returns:
        --------
        bool
            True if coordinates are connected
        """
        idx1 = self._coord_to_index(coord1)
        idx2 = self._coord_to_index(coord2)
        return self.uf.connected(idx1, idx2)
    
    def find_coord(self, coord: tuple[int, ...]) -> tuple[int, ...]:
        """
        Find root coordinate of component containing given coordinate.
        
        Parameters:
        -----------
        coord : tuple of int
            Grid coordinate
            
        Returns:
        --------
        tuple of int
            Root coordinate of component
        """
        idx = self._coord_to_index(coord)
        root_idx = self.uf.find(idx)
        return self._index_to_coord(root_idx)
    
    def get_component_coords(self, coord: tuple[int, ...]) -> list[tuple[int, ...]]:
        """
        Get all coordinates in same component as given coordinate.
        
        Parameters:
        -----------
        coord : tuple of int
            Grid coordinate
            
        Returns:
        --------
        list[tuple of int]
            List of coordinates in same component
        """
        idx = self._coord_to_index(coord)
        component_indices = self.uf.get_component_containing(idx)
        return [self._index_to_coord(idx) for idx in component_indices]
    
    def get_all_components_coords(self) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
        """
        Get all components as coordinate lists.
        
        Returns:
        --------
        dict[tuple of int, list[tuple of int]]
            Dictionary mapping root coordinates to component coordinate lists
        """
        components = self.uf.get_components()
        coord_components = {}
        
        for root_idx, indices in components.items():
            root_coord = self._index_to_coord(root_idx)
            coord_list = [self._index_to_coord(idx) for idx in indices]
            coord_components[root_coord] = coord_list
        
        return coord_components
    
    def union_neighbors(self, coord: tuple[int, ...], neighbors: list[tuple[int, ...]], data: np.ndarray, condition_func: callable = None):
        """
        Union coordinate with its neighbors based on condition.
        
        Parameters:
        -----------
        coord : tuple of int
            Central coordinate
        neighbors : list of tuple of int
            Neighbor coordinates
        data : np.ndarray
            Data array for condition checking
        condition_func : callable, optional
            Function to test if union should occur: condition_func(coord, neighbor, data) -> bool
        """
        if condition_func is None:
            # Default: union if values are equal
            def condition_func(c1, c2, d):
                return d[c1] == d[c2]
        
        for neighbor in neighbors:
            # Check bounds
            if all(0 <= neighbor[i] < self.shape[i] for i in range(self.ndim)):
                if condition_func(coord, neighbor, data):
                    self.union_coords(coord, neighbor)
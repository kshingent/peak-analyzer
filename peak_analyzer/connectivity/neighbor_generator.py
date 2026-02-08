from typing import Iterator
from functools import lru_cache

@lru_cache(maxsize=16384)
def get_neighbors(
    center: tuple[int, ...], 
    shape: tuple[int, ...], 
    offsets: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """
    Compute and cache valid neighbor coordinates using hashable arguments.
    Minimizes re-computation cost during topographic priority search (BFS/Heap)
    when the search front revisits the same coordinates.
    """
    valid_neighbors = []
    
    for offset in offsets:
        # N-dimensional coordinate addition
        neighbor = tuple(c + o for c, o in zip(center, offset))
        
        # Boundary condition check for all dimensions
        if all(0 <= n < s for n, s in zip(neighbor, shape)):
            valid_neighbors.append(neighbor)
            
    return tuple(valid_neighbors)

def get_iterator(
    center: tuple[int, ...], 
    shape: tuple[int, ...], 
    offsets: tuple[tuple[int, ...], ...]
) -> Iterator[tuple[int, ...]]:
    """Iterator for seamless deployment within search loops."""
    return iter(get_neighbors(center, shape, offsets))
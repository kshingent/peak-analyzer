"""
Unit tests for coordinate system components.

Tests the separation between index space (ijk...) and coordinate space (xyz...)
as defined in the coordinate system architecture.
"""

import sys
import os
import pytest
import numpy as np

# Add the package root to the path to import modules directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from peak_analyzer.coordinate_system.coordinate_mapping import (
    CoordinateMapping, 
    create_isotropic_mapping, 
    create_anisotropic_mapping,
    create_physical_mapping
)
from peak_analyzer.coordinate_system.grid_manager import GridManager


class TestCoordinateMapping:
    """Test CoordinateMapping functionality."""
    
    def test_basic_coordinate_mapping_creation(self):
        """Test basic coordinate mapping creation."""
        mapping = CoordinateMapping(
            scale=(1.0, 2.0, 0.5),
            origin=(0.0, 10.0, -5.0),
            axis_labels=('x', 'y', 'z'),
            units=('mm', 'mm', 'mm')
        )
        
        assert mapping.ndim == 3
        assert mapping.scale == (1.0, 2.0, 0.5)
        assert mapping.origin == (0.0, 10.0, -5.0)
        assert mapping.axis_labels == ('x', 'y', 'z')
        assert mapping.units == ('mm', 'mm', 'mm')
    
    def test_mapping_with_defaults(self):
        """Test coordinate mapping with default values."""
        mapping = CoordinateMapping(scale=(2.0, 1.5))
        
        assert mapping.ndim == 2
        assert mapping.scale == (2.0, 1.5)
        assert mapping.origin == (0.0, 0.0)
        assert mapping.axis_labels == ('x', 'y')
        assert mapping.units == ('unit', 'unit')
    
    def test_index_to_coordinate_conversion(self):
        """Test conversion from index space to coordinate space."""
        mapping = CoordinateMapping(
            scale=(1.0, 2.0, 0.5),
            origin=(0.0, 10.0, -5.0)
        )
        
        # Test single point conversion
        indices = (0, 0, 0)
        coords = mapping.index_to_coordinate(indices)
        assert coords == (0.0, 10.0, -5.0)
        
        # Test with non-zero indices
        indices = (10, 5, 20)
        coords = mapping.index_to_coordinate(indices)
        expected = (10.0, 20.0, 5.0)  # (0+10*1.0, 10+5*2.0, -5+20*0.5)
        assert coords == expected
        
        # Test numpy array conversion
        indices_array = np.array([[0, 0, 0], [10, 5, 20]])
        coords_array = mapping.index_to_coordinate(indices_array)
        expected_array = np.array([[0.0, 10.0, -5.0], [10.0, 20.0, 5.0]])
        np.testing.assert_array_equal(coords_array, expected_array)
    
    def test_coordinate_to_index_conversion_nearest(self):
        """Test conversion from coordinate space to index space with nearest rounding."""
        mapping = CoordinateMapping(
            scale=(1.0, 2.0, 0.5),
            origin=(0.0, 10.0, -5.0)
        )
        
        # Test exact coordinate conversion
        coords = (10.0, 20.0, 5.0)
        indices = mapping.coordinate_to_index(coords, round_mode='nearest')
        assert indices == (10, 5, 20)
        
        # Test fractional coordinates - should round to nearest
        coords = (10.4, 19.7, 5.2)  # Should round to (10, 5, 20)
        indices = mapping.coordinate_to_index(coords, round_mode='nearest')
        assert indices == (10, 5, 20)
        
        coords = (10.6, 20.3, 4.8)  # Should round to (11, 5, 20)
        indices = mapping.coordinate_to_index(coords, round_mode='nearest')
        assert indices == (11, 5, 20)
    
    def test_coordinate_to_index_conversion_floor_ceil(self):
        """Test floor and ceil rounding modes."""
        mapping = CoordinateMapping(scale=(1.0, 1.0), origin=(0.0, 0.0))
        
        coords = (2.7, 3.2)
        
        # Floor rounding
        indices_floor = mapping.coordinate_to_index(coords, round_mode='floor')
        assert indices_floor == (2, 3)
        
        # Ceil rounding
        indices_ceil = mapping.coordinate_to_index(coords, round_mode='ceil')
        assert indices_ceil == (3, 4)
    
    def test_anisotropic_scaling(self):
        """Test anisotropic scaling (different scales per dimension)."""
        # Create mapping with different scale factors
        mapping = create_anisotropic_mapping(
            scale=[1.0, 1.0, 0.5],  # Z-axis has half resolution
            origin=[0.0, 0.0, 0.0]
        )
        
        # Test that scaling is correctly applied
        indices = (10, 10, 10)
        coords = mapping.index_to_coordinate(indices)
        assert coords == (10.0, 10.0, 5.0)  # Z should be scaled by 0.5
        
        # Test reverse conversion
        coords = (10.0, 10.0, 5.0)
        indices_back = mapping.coordinate_to_index(coords)
        assert indices_back == (10, 10, 10)
    
    def test_coordinate_bounds(self):
        """Test coordinate bounds calculation."""
        mapping = CoordinateMapping(
            scale=(2.0, 0.5),
            origin=(10.0, -5.0)
        )
        
        shape = (5, 10)  # 5x10 grid
        min_coords, max_coords = mapping.get_coordinate_bounds(shape)
        
        # Min bounds at index (0, 0)
        assert min_coords == (10.0, -5.0)
        
        # Max bounds at index (4, 9)
        expected_max = (10.0 + 4*2.0, -5.0 + 9*0.5)  # (18.0, -0.5)
        assert max_coords == expected_max
    
    def test_distance_calculation(self):
        """Test Euclidean distance calculation."""
        mapping = CoordinateMapping(scale=(1.0, 1.0, 1.0))
        
        coord1 = (0.0, 0.0, 0.0)
        coord2 = (3.0, 4.0, 0.0)
        
        distance = mapping.calculate_distance(coord1, coord2)
        expected = 5.0  # 3-4-5 triangle
        assert abs(distance - expected) < 1e-10
    
    def test_voxel_volume_calculation(self):
        """Test voxel volume calculation."""
        mapping = CoordinateMapping(scale=(2.0, 1.5, 0.8))
        
        volume = mapping.get_voxel_volume()
        expected = 2.0 * 1.5 * 0.8
        assert abs(volume - expected) < 1e-10
    
    def test_roundtrip_conversion_consistency(self):
        """Test that index->coordinate->index conversion is consistent."""
        mapping = CoordinateMapping(
            scale=(1.5, 0.8, 2.2),
            origin=(5.0, -3.0, 1.0)
        )
        
        # Test multiple points
        test_indices = [(0, 0, 0), (5, 10, 3), (100, 50, 25)]
        
        for original_indices in test_indices:
            coords = mapping.index_to_coordinate(original_indices)
            recovered_indices = mapping.coordinate_to_index(coords, round_mode='nearest')
            assert recovered_indices == original_indices
    
    def test_validation_errors(self):
        """Test validation errors for invalid parameters."""
        # Test invalid scale factors
        with pytest.raises(ValueError, match="All scale factors must be positive"):
            CoordinateMapping(scale=(1.0, 0.0))  # Zero scale
        
        with pytest.raises(ValueError, match="All scale factors must be positive"):
            CoordinateMapping(scale=(1.0, -1.0))  # Negative scale
            
        # Test mismatched dimensions
        with pytest.raises(ValueError, match="Origin dimensions .* must match scale dimensions"):
            CoordinateMapping(scale=(1.0, 2.0), origin=(1.0, 2.0, 3.0))
            
        with pytest.raises(ValueError, match="Axis labels length .* must match dimensions"):
            CoordinateMapping(scale=(1.0, 2.0), axis_labels=('x', 'y', 'z'))
            
        with pytest.raises(ValueError, match="Units length .* must match dimensions"):
            CoordinateMapping(scale=(1.0, 2.0), units=('mm', 'mm', 'mm'))
    
    def test_type_validation(self):
        """Test type validation for conversion methods."""
        mapping = CoordinateMapping(scale=(1.0, 2.0))
        
        # Test dimension mismatch
        with pytest.raises(ValueError, match="Index dimensions .* must match mapping dimensions"):
            mapping.index_to_coordinate((1, 2, 3))  # 3D indices for 2D mapping
        
        with pytest.raises(ValueError, match="Coordinate dimensions .* must match mapping dimensions"):
            mapping.coordinate_to_index((1.0, 2.0, 3.0))  # 3D coords for 2D mapping
        
        # Test invalid types
        with pytest.raises(TypeError, match="Indices must be tuple or numpy array"):
            mapping.index_to_coordinate([1, 2])  # List instead of tuple/array
            
        with pytest.raises(TypeError, match="Coordinates must be tuple or numpy array"):
            mapping.coordinate_to_index([1.0, 2.0])  # List instead of tuple/array
        
        # Test invalid round mode
        with pytest.raises(ValueError, match="Unknown round_mode"):
            mapping.coordinate_to_index((1.0, 2.0), round_mode='invalid')


class TestGridManager:
    """Test GridManager functionality."""
    
    def test_grid_manager_creation(self):
        """Test GridManager creation and basic properties."""
        mapping = CoordinateMapping(
            scale=(1.0, 2.0),
            origin=(0.0, 10.0)
        )
        shape = (10, 20)
        
        grid = GridManager(shape, mapping)
        
        assert grid.shape == (10, 20)
        assert grid.ndim == 2
        assert grid.size == 200
        assert grid.mapping == mapping
    
    def test_grid_creation_validation(self):
        """Test validation during grid creation."""
        mapping = CoordinateMapping(scale=(1.0, 2.0))
        
        # Test dimension mismatch
        with pytest.raises(ValueError, match="Shape dimensions .* must match mapping dimensions"):
            GridManager((10, 20, 5), mapping)  # 3D shape for 2D mapping
    
    def test_grid_coordinate_bounds(self):
        """Test grid coordinate bounds calculation."""
        mapping = CoordinateMapping(scale=(0.5, 1.5), origin=(2.0, -1.0))
        grid = GridManager((100, 50), mapping)
        
        min_coords, max_coords = grid.get_coordinate_bounds()
        
        # Min at index (0, 0): (2.0, -1.0)
        assert min_coords == (2.0, -1.0)
        
        # Max at index (99, 49): (2.0 + 99*0.5, -1.0 + 49*1.5)
        expected_max = (2.0 + 99*0.5, -1.0 + 49*1.5)
        assert max_coords == expected_max
    
    def test_valid_index_checking(self):
        """Test index validity checking."""
        mapping = CoordinateMapping(scale=(1.0, 1.0), origin=(0.0, 0.0))
        grid = GridManager((10, 5), mapping)
        
        # Valid indices
        assert grid.is_valid_index((0, 0))
        assert grid.is_valid_index((9, 4))
        assert grid.is_valid_index((5, 2))
        
        # Invalid indices
        assert not grid.is_valid_index((-1, 0))
        assert not grid.is_valid_index((10, 0))
        assert not grid.is_valid_index((0, 5))
        assert not grid.is_valid_index((5, -1))
        
        # Test with numpy array
        indices_array = np.array([[0, 0], [9, 4], [10, 0], [-1, 2]])
        valid_mask = grid.is_valid_index(indices_array)
        expected = np.array([True, True, False, False])
        np.testing.assert_array_equal(valid_mask, expected)
    
    def test_valid_coordinate_checking(self):
        """Test coordinate validity checking."""
        mapping = CoordinateMapping(scale=(2.0, 0.5), origin=(1.0, 3.0))
        grid = GridManager((5, 10), mapping)
        
        # Valid coordinates
        assert grid.is_valid_coordinate((1.0, 3.0))  # Index (0, 0)
        assert grid.is_valid_coordinate((9.0, 7.5))  # Index (4, 9)
        
        # Invalid coordinates
        assert not grid.is_valid_coordinate((0.0, 3.0))  # Before grid
        assert not grid.is_valid_coordinate((11.0, 3.0))  # After grid
    
    def test_index_clipping(self):
        """Test index clipping to grid bounds."""
        mapping = CoordinateMapping(scale=(1.0, 1.0), origin=(0.0, 0.0))
        grid = GridManager((10, 5), mapping)
        
        # Test clipping out-of-bounds indices
        assert grid.clip_indices((-1, 2)) == (0, 2)
        assert grid.clip_indices((12, 3)) == (9, 3)
        assert grid.clip_indices((5, -2)) == (5, 0)
        assert grid.clip_indices((3, 8)) == (3, 4)
        
        # Test with numpy array
        indices_array = np.array([[-1, 2], [12, 3], [5, -2], [3, 8]])
        clipped = grid.clip_indices(indices_array)
        expected = np.array([[0, 2], [9, 3], [5, 0], [3, 4]])
        np.testing.assert_array_equal(clipped, expected)
    
    def test_coordinate_clipping(self):
        """Test coordinate clipping to grid bounds."""
        mapping = CoordinateMapping(scale=(2.0, 1.0), origin=(0.0, 5.0))
        grid = GridManager((5, 3), mapping)  # Coordinates: x=[0,8], y=[5,7]
        
        # Test clipping
        assert grid.clip_coordinates((-1.0, 6.0)) == (0.0, 6.0)
        assert grid.clip_coordinates((10.0, 6.0)) == (8.0, 6.0)
        assert grid.clip_coordinates((4.0, 4.0)) == (4.0, 5.0)
        assert grid.clip_coordinates((4.0, 8.0)) == (4.0, 7.0)
    
    def test_coordinate_mesh_generation(self):
        """Test coordinate mesh generation."""
        mapping = CoordinateMapping(scale=(1.0, 2.0), origin=(5.0, 10.0))
        grid = GridManager((3, 2), mapping)
        
        coord_mesh = grid.generate_coordinate_mesh()
        assert len(coord_mesh) == 2  # 2D
        
        # Check shapes
        assert coord_mesh[0].shape == (3, 2)
        assert coord_mesh[1].shape == (3, 2)
        
        # Check values at specific positions
        # At index (0, 0): coordinates should be (5.0, 10.0)
        assert coord_mesh[0][0, 0] == 5.0
        assert coord_mesh[1][0, 0] == 10.0
        
        # At index (2, 1): coordinates should be (7.0, 12.0)
        assert coord_mesh[0][2, 1] == 7.0
        assert coord_mesh[1][2, 1] == 12.0
    
    def test_index_mesh_generation(self):
        """Test index mesh generation."""
        mapping = CoordinateMapping(scale=(1.0, 1.0))
        grid = GridManager((3, 2), mapping)
        
        index_mesh = grid.generate_index_mesh()
        assert len(index_mesh) == 2  # 2D
        
        # Check shapes
        assert index_mesh[0].shape == (3, 2)
        assert index_mesh[1].shape == (3, 2)
        
        # Check values
        expected_i = np.array([[0, 0], [1, 1], [2, 2]])
        expected_j = np.array([[0, 1], [0, 1], [0, 1]])
        np.testing.assert_array_equal(index_mesh[0], expected_i)
        np.testing.assert_array_equal(index_mesh[1], expected_j)
    
    def test_grid_coordinates_flattened(self):
        """Test flattened grid coordinates."""
        mapping = CoordinateMapping(scale=(2.0, 1.0), origin=(0.0, 5.0))
        grid = GridManager((2, 3), mapping)
        
        coords = grid.get_grid_coordinates()
        
        # Should have shape (6, 2) for 2x3 grid in 2D
        assert coords.shape == (6, 2)
        
        # Check specific coordinates
        expected = np.array([
            [0.0, 5.0],  # (0,0)
            [0.0, 6.0],  # (0,1)
            [0.0, 7.0],  # (0,2)
            [2.0, 5.0],  # (1,0)
            [2.0, 6.0],  # (1,1)
            [2.0, 7.0],  # (1,2)
        ])
        np.testing.assert_array_equal(coords, expected)


class TestCoordinateSystemIntegration:
    """Integration tests for coordinate system components."""
    
    def test_scale_preservation_in_distance_calculation(self):
        """Test that scale factors are correctly preserved in distance calculations."""
        # Create anisotropic mapping with different scales
        mapping = CoordinateMapping(
            scale=(1.0, 2.0, 0.5),  # Different scale per dimension
            origin=(0.0, 0.0, 0.0)
        )
        
        # Calculate distance between two coordinates
        coord1 = (0.0, 0.0, 0.0)
        coord2 = (1.0, 2.0, 0.5)
        
        distance = mapping.calculate_distance(coord1, coord2)
        expected = np.sqrt(1.0**2 + 2.0**2 + 0.5**2)  # Euclidean distance
        assert abs(distance - expected) < 1e-10
    
    def test_fractional_coordinate_handling(self):
        """Test handling of fractional coordinates in conversions."""
        mapping = CoordinateMapping(scale=(0.1, 0.1), origin=(0.0, 0.0))
        
        # Test coordinates that don't fall exactly on grid points
        coords = (0.15, 0.27)
        
        # Different rounding modes should give different results
        nearest = mapping.coordinate_to_index(coords, round_mode='nearest')
        floor = mapping.coordinate_to_index(coords, round_mode='floor')
        ceil = mapping.coordinate_to_index(coords, round_mode='ceil')
        
        # coords (0.15, 0.27) -> indices (1.5, 2.7)
        assert nearest == (2, 3)  # round(1.5, 2.7)
        assert floor == (1, 2)    # floor(1.5, 2.7)
        assert ceil == (2, 3)     # ceil(1.5, 2.7)
    
    def test_physical_mapping_creation(self):
        """Test creation of physical coordinate mappings."""
        mapping = create_physical_mapping(
            physical_scale=[0.1, 0.1, 0.5],  # mm per pixel
            physical_units=['mm', 'mm', 'mm'],
            axis_labels=['x', 'y', 'z'],
            origin=[10.0, 20.0, 0.0]
        )
        
        assert mapping.scale == (0.1, 0.1, 0.5)
        assert mapping.units == ('mm', 'mm', 'mm')
        assert mapping.axis_labels == ('x', 'y', 'z')
        assert mapping.origin == (10.0, 20.0, 0.0)
        
        # Test conversion
        indices = (10, 50, 6)
        coords = mapping.index_to_coordinate(indices)
        expected = (11.0, 25.0, 3.0)  # (10+10*0.1, 20+50*0.1, 0+6*0.5)
        assert coords == expected
    
    def test_isotropic_mapping_creation(self):
        """Test creation of isotropic mappings."""
        mapping = create_isotropic_mapping(scale=2.0, origin=5.0, ndim=3)
        
        assert mapping.scale == (2.0, 2.0, 2.0)
        assert mapping.origin == (5.0, 5.0, 5.0)
        assert mapping.ndim == 3
    
    def test_coordinate_round_trip_with_noise(self):
        """Test coordinate round-trip conversion with numerical noise."""
        mapping = CoordinateMapping(scale=(1.0/3.0, 1.0/7.0), origin=(0.0, 0.0))
        
        # Test indices that might introduce floating point errors
        test_indices = [(3, 7), (9, 21), (12, 14)]
        
        for original_indices in test_indices:
            # Forward conversion
            coords = mapping.index_to_coordinate(original_indices)
            
            # Add small numerical noise
            noisy_coords = (
                coords[0] + 1e-14,
                coords[1] - 1e-14
            )
            
            # Backward conversion should still recover original indices
            recovered_indices = mapping.coordinate_to_index(noisy_coords, round_mode='nearest')
            assert recovered_indices == original_indices
    
    def test_large_grid_operations(self):
        """Test operations on larger grids for performance validation.""" 
        mapping = CoordinateMapping(scale=(0.01, 0.01), origin=(0.0, 0.0))
        grid = GridManager((1000, 1000), mapping)
        
        # Test bounds calculation
        min_coords, max_coords = grid.get_coordinate_bounds()
        assert min_coords == (0.0, 0.0)
        expected_max = (9.99, 9.99)  # (999*0.01, 999*0.01)
        np.testing.assert_allclose(max_coords, expected_max, rtol=1e-12)
        
        # Test random coordinate validation
        test_coords = [
            (5.0, 5.0),    # Valid
            (10.0, 5.0),   # Invalid (x too large)
            (-1.0, 5.0),   # Invalid (x too small)
        ]
        
        expected_valid = [True, False, False]
        for i, coord in enumerate(test_coords):
            assert grid.is_valid_coordinate(coord) == expected_valid[i]


@pytest.mark.parametrize("scale,origin,test_indices", [
    # Isotropic scaling
    ((1.0, 1.0), (0.0, 0.0), [(0, 0), (5, 3), (10, 10)]),
    # Anisotropic scaling
    ((2.0, 0.5), (1.0, -2.0), [(0, 0), (3, 8), (7, 15)]),
    # High resolution
    ((0.1, 0.1, 0.1), (0.0, 0.0, 0.0), [(0, 0, 0), (100, 200, 50)]),
])
def test_parametrized_coordinate_conversions(scale, origin, test_indices):
    """Parametrized test for various coordinate conversion scenarios."""
    mapping = CoordinateMapping(scale=scale, origin=origin)
    
    for indices in test_indices:
        # Forward conversion
        coords = mapping.index_to_coordinate(indices)
        
        # Backward conversion
        recovered_indices = mapping.coordinate_to_index(coords, round_mode='nearest')
        
        # Should recover original indices
        assert recovered_indices == indices
        
        # Check that coordinates are calculated correctly
        expected_coords = tuple(
            origin[i] + indices[i] * scale[i] 
            for i in range(len(indices))
        )
        assert coords == expected_coords


@pytest.mark.parametrize("round_mode,coords,expected", [
    ('nearest', (2.4, 3.6), (2, 4)),
    ('nearest', (2.5, 3.5), (2, 4)),  # Note: Python's round() uses banker's rounding for .5
    ('nearest', (2.6, 3.4), (3, 3)),
    ('floor', (2.7, 3.2), (2, 3)),
    ('floor', (2.1, 3.9), (2, 3)),
    ('ceil', (2.1, 3.2), (3, 4)),
    ('ceil', (2.9, 3.1), (3, 4)),
])
def test_parametrized_rounding_modes(round_mode, coords, expected):
    """Parametrized test for different rounding modes."""
    mapping = CoordinateMapping(scale=(1.0, 1.0), origin=(0.0, 0.0))
    
    result = mapping.coordinate_to_index(coords, round_mode=round_mode)
    assert result == expected


class TestSpecificRequirements:
    """Tests specifically addressing the requirements from README_japanese.md."""
    
    def test_scale_reflection_in_coordinate_transformation(self):
        """
        検証ポイント1: GridManager または CoordinateMapping が、
        スケール設定（scale）を正しく反映して座標変換できているか。
        """
        # Create mapping with specific scale factors
        scale = (2.5, 0.8, 1.2)
        origin = (10.0, -5.0, 2.0)
        mapping = CoordinateMapping(scale=scale, origin=origin)
        
        # Test that scale is correctly applied in forward transformation
        indices = (4, 10, 5)
        coords = mapping.index_to_coordinate(indices)
        
        # Manual calculation: coord = origin + index * scale
        expected_coords = (
            10.0 + 4 * 2.5,   # 20.0
            -5.0 + 10 * 0.8,  # 3.0
            2.0 + 5 * 1.2     # 8.0
        )
        assert coords == expected_coords
        
        # Test that scale is correctly applied in reverse transformation
        recovered_indices = mapping.coordinate_to_index(coords, round_mode='nearest')
        assert recovered_indices == indices
        
        # Test with GridManager
        grid = GridManager((20, 30, 15), mapping)
        grid_coords = grid.index_to_coordinate(indices)
        assert grid_coords == coords
        
        # Verify scale is preserved in distance calculations
        coord1 = (10.0, -5.0, 2.0)  # Origin
        coord2 = (12.5, -4.2, 3.2)  # Origin + (1*2.5, 1*0.8, 1*1.2)
        distance = mapping.calculate_distance(coord1, coord2)
        expected_distance = np.sqrt(2.5**2 + 0.8**2 + 1.2**2)
        assert abs(distance - expected_distance) < 1e-10
    
    def test_nearest_index_for_fractional_coordinates(self):
        """
        検証ポイント2: 逆変換（座標からインデックス）の際、端数が出た場合に
        最も近いインデックスを正しく返しているか。
        """
        # Create mapping with non-unit scale to ensure fractional results
        mapping = CoordinateMapping(scale=(0.7, 1.3, 0.4), origin=(1.5, -2.1, 0.8))
        
        # Test coordinates that will result in fractional indices
        test_cases = [
            # (coordinates, expected_nearest_indices)
            ((2.55, -0.45), (1, 1)),    # Should round (1.5, 1.27) to (2, 1) 
            ((2.91, -0.84, 2.0), (2, 1, 3)),  # Should round (2.01, 1.26, 3.0) to (2, 1, 3)
            ((1.85, -1.14), (1, 1)),    # Should round (0.5, 0.74) to (1, 1)
            ((1.15, -2.75), (0, 0)),    # Should round (-0.5, -0.5) to (0, 0) 
        ]
        
        for coords, expected in test_cases:
            # Test nearest rounding
            result = mapping.coordinate_to_index(coords, round_mode='nearest')
            assert result == expected, f"For coords {coords}, expected {expected}, got {result}"
            
            # Verify that the computed indices, when converted back, 
            # are indeed the closest to the original coordinates
            back_coords = mapping.index_to_coordinate(result)
            distance_to_result = mapping.calculate_distance(coords, back_coords)
            
            # Test neighboring indices to confirm this is indeed the nearest
            for dim in range(len(expected)):
                for offset in [-1, 1]:
                    neighbor_indices = list(expected)
                    neighbor_indices[dim] += offset
                    neighbor_indices = tuple(neighbor_indices)
                    
                    neighbor_coords = mapping.index_to_coordinate(neighbor_indices)
                    distance_to_neighbor = mapping.calculate_distance(coords, neighbor_coords)
                    
                    # The chosen index should be closer or equal distance
                    assert distance_to_result <= distance_to_neighbor + 1e-10, (
                        f"Neighbor {neighbor_indices} is closer to {coords} than chosen {expected}"
                    )
    
    def test_edge_case_rounding_behavior(self):
        """Test rounding behavior for edge cases (exactly .5 fractional parts)."""
        mapping = CoordinateMapping(scale=(1.0, 1.0), origin=(0.0, 0.0))
        
        # Test coordinates that result in exactly .5 fractional indices
        # Python's round() uses "round half to even" (banker's rounding)
        test_cases = [
            ((2.5, 3.5), (2, 4)),  # round(2.5) = 2 (even), round(3.5) = 4 (even)
            ((1.5, 4.5), (2, 4)),  # round(1.5) = 2 (even), round(4.5) = 4 (even)
            ((3.5, 5.5), (4, 6)),  # round(3.5) = 4 (even), round(5.5) = 6 (even)
        ]
        
        for coords, expected in test_cases:
            result = mapping.coordinate_to_index(coords, round_mode='nearest')
            assert result == expected
    
    def test_architecture_separation_verification(self):
        """
        Verify that the architecture properly separates index space (computational) 
        from coordinate space (user-facing) as defined in README_japanese.md.
        """
        # Create a mapping that represents the separation
        # Index space: integer grid positions (i, j, k)
        # Coordinate space: real-world positions (x, y, z) with physical meaning
        
        physical_mapping = create_physical_mapping(
            physical_scale=[0.5, 0.5, 1.0],  # mm per voxel
            physical_units=['mm', 'mm', 'mm'],
            axis_labels=['x', 'y', 'z'],
            origin=[100.0, 200.0, 50.0]  # mm
        )
        
        # Verify that index space operations work with integers
        index_position = (10, 20, 5)
        coordinate_position = physical_mapping.index_to_coordinate(index_position)
        
        # Verify coordinate space has physical meaning
        expected_coordinates = (
            100.0 + 10 * 0.5,  # 105.0 mm
            200.0 + 20 * 0.5,  # 210.0 mm
            50.0 + 5 * 1.0     # 55.0 mm
        )
        assert coordinate_position == expected_coordinates
        
        # Verify that the mapping preserves units and labels
        assert physical_mapping.units == ('mm', 'mm', 'mm')
        assert physical_mapping.axis_labels == ('x', 'y', 'z')
        
        # Verify that GridManager operates consistently with this separation
        grid = GridManager((50, 40, 30), physical_mapping)
        
        # Index space: discrete grid positions
        assert grid.shape == (50, 40, 30)
        assert grid.size == 50 * 40 * 30
        
        # Coordinate space: continuous physical positions
        min_coords, max_coords = grid.get_coordinate_bounds()
        assert min_coords == (100.0, 200.0, 50.0)
        assert max_coords == (
            100.0 + 49 * 0.5,  # 124.5 mm
            200.0 + 39 * 0.5,  # 219.5 mm
            50.0 + 29 * 1.0    # 79.0 mm
        )
        
        # Verify that transformations preserve the separation
        arbitrary_coords = (115.5, 207.5, 63.0)
        indices = grid.coordinate_to_index(arbitrary_coords, round_mode='nearest')
        back_coords = grid.index_to_coordinate(indices)
        
        # The round-trip should maintain the architectural separation:
        # Index space contains integers, coordinate space contains physical values
        assert all(isinstance(i, (int, np.integer)) for i in indices)
        assert all(isinstance(c, (float, np.floating)) for c in back_coords)
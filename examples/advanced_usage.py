"""
Advanced usage examples demonstrating 3D data and feature extraction.
"""

import numpy as np
from peak_analyzer import PeakAnalyzer


def create_3d_sample_data():
    """Create sample 3D data for testing."""
    # Create a 3D grid
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    z = np.linspace(-3, 3, 50)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Create multiple 3D peaks
    peak1 = 2 * np.exp(-((X - 1)**2 + (Y - 1)**2 + (Z - 1)**2) / 1.0)
    peak2 = 1.5 * np.exp(-((X + 1)**2 + (Y)**2 + (Z + 1)**2) / 0.5)
    peak3 = 1 * np.exp(-((X)**2 + (Y + 1)**2 + (Z)**2) / 2.0)
    
    data = peak1 + peak2 + peak3
    
    return data


def demonstrate_3d_detection():
    """Demonstrate peak detection in 3D data."""
    print("=== 3D Peak Detection ===")
    
    data = create_3d_sample_data()
    print(f"Data shape: {data.shape}")
    
    analyzer = PeakAnalyzer(
        strategy="batch_heap",  # Better for 3D smooth data
        min_prominence=0.3
    )
    
    peaks = analyzer.find_peaks(data)
    print(f"Found {len(peaks)} peaks in 3D data")
    
    if len(peaks) > 0:
        # Extract 3D coordinates
        x_coords = peaks.get_feature('centroid_x')
        y_coords = peaks.get_feature('centroid_y')
        # For 3D, we need to manually extract z coordinates
        z_coords = []
        for i in range(len(peaks.peak_regions)):
            if len(peaks.peak_regions[i].centroid) > 2:
                z_coords.append(peaks.peak_regions[i].centroid[2])
            else:
                z_coords.append(0)
        
        heights = peaks.get_feature('height')
        
        print("Peak locations (x, y, z, height):")
        for i in range(len(x_coords)):
            print(f"  Peak {i+1}: ({x_coords[i]:.2f}, {y_coords[i]:.2f}, "
                  f"{z_coords[i]:.2f}, height={heights[i]:.2f})")


def demonstrate_feature_extraction():
    """Demonstrate advanced feature extraction."""
    print("\n=== Feature Extraction ===")
    
    # Create 2D data for easier visualization
    x = np.linspace(-4, 4, 80)
    y = np.linspace(-4, 4, 80)
    X, Y = np.meshgrid(x, y)
    
    # Create peaks with different characteristics
    sharp_peak = 3 * np.exp(-((X - 1)**2 + (Y - 1)**2) / 0.1)
    broad_peak = 2 * np.exp(-((X + 1)**2 + (Y - 1)**2) / 3.0)
    medium_peak = 1.8 * np.exp(-((X)**2 + (Y + 1)**2) / 0.8)
    
    data = sharp_peak + broad_peak + medium_peak
    
    analyzer = PeakAnalyzer(min_prominence=0.2)
    peaks = analyzer.find_peaks(data)
    
    print(f"Analyzing {len(peaks)} peaks...")
    
    if len(peaks) > 0:
        # Get all available features
        heights = peaks.get_feature('height')
        areas = peaks.get_feature('area')
        prominences = peaks.get_feature('prominence')
        isolations = peaks.get_feature('isolation')
        
        print("\nPeak Analysis:")
        print("Peak | Height | Area | Prominence | Isolation")
        print("-" * 50)
        
        for i in range(len(heights)):
            print(f"  {i+1:2d} | {heights[i]:6.2f} | {areas[i]:4d} | "
                  f"{prominences[i]:10.2f} | {isolations[i]:9.2f}")


def demonstrate_strategy_comparison():
    """Compare different detection strategies."""
    print("\n=== Strategy Comparison ===")
    
    # Create test data
    x = np.linspace(-3, 3, 60)
    y = np.linspace(-3, 3, 60)
    X, Y = np.meshgrid(x, y)
    
    # Smooth terrain with gradual peaks
    data = (np.sin(2 * X) * np.cos(2 * Y) + 
            0.5 * np.sin(4 * X) * np.cos(Y) + 
            0.3 * np.random.random((60, 60)))
    
    strategies = ["independent", "batch_heap", "auto"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        
        analyzer = PeakAnalyzer(
            strategy=strategy,
            min_prominence=0.1
        )
        
        peaks = analyzer.find_peaks(data)
        print(f"  Found {len(peaks)} peaks")
        
        if len(peaks) > 0:
            heights = peaks.get_feature('height')
            avg_height = np.mean(heights)
            max_height = np.max(heights)
            print(f"  Average height: {avg_height:.3f}")
            print(f"  Maximum height: {max_height:.3f}")


if __name__ == "__main__":
    # Run 3D demonstration
    demonstrate_3d_detection()
    
    # Feature extraction
    demonstrate_feature_extraction()
    
    # Strategy comparison
    demonstrate_strategy_comparison()
    
    print("\nAdvanced examples completed!")
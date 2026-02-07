"""
Basic usage example for peak-analyzer.
"""

import numpy as np
import matplotlib.pyplot as plt
from peak_analyzer import PeakAnalyzer


def create_sample_data():
    """Create sample 2D data with multiple peaks."""
    # Create a 2D grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create multiple peaks with different characteristics
    peak1 = 3 * np.exp(-((X - 1)**2 + (Y - 1)**2) / 0.5)  # Sharp peak
    peak2 = 2 * np.exp(-((X + 2)**2 + (Y - 2)**2) / 2.0)  # Broad peak
    peak3 = 1.5 * np.exp(-((X - 2)**2 + (Y + 1)**2) / 0.3)  # Medium peak
    
    # Add some noise
    noise = 0.1 * np.random.random((100, 100))
    
    data = peak1 + peak2 + peak3 + noise
    
    return data, X, Y


def basic_peak_detection():
    """Demonstrate basic peak detection functionality."""
    print("Creating sample data...")
    data, X, Y = create_sample_data()
    
    print("Initializing PeakAnalyzer...")
    analyzer = PeakAnalyzer(
        strategy="auto",
        min_prominence=0.5,
        min_area=1
    )
    
    print("Finding peaks...")
    peaks = analyzer.find_peaks(data)
    
    print(f"Found {len(peaks)} peaks")
    
    # Convert to DataFrame for easier inspection
    peak_df = peaks.to_dataframe()
    print("\nPeak details:")
    print(peak_df)
    
    return data, peaks, X, Y


def plot_results(data, peaks, X, Y):
    """Plot the results."""
    plt.figure(figsize=(12, 5))
    
    # Plot original data
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, data, levels=20, cmap='viridis')
    plt.colorbar(label='Height')
    plt.title('Original Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot detected peaks
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, data, levels=20, cmap='viridis', alpha=0.7)
    
    # Mark detected peaks
    if len(peaks) > 0:
        centroids_x = peaks.get_feature('centroid_x')
        centroids_y = peaks.get_feature('centroid_y')
        heights = peaks.get_feature('height')
        
        scatter = plt.scatter(centroids_x, centroids_y, c=heights, 
                            s=100, cmap='Reds', edgecolors='black',
                            linewidth=2, marker='^')
        plt.colorbar(scatter, label='Peak Height')
    
    plt.title('Detected Peaks')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.tight_layout()
    plt.show()


def demonstrate_filtering():
    """Demonstrate peak filtering capabilities."""
    print("\n=== Filtering Demonstration ===")
    
    data, X, Y = create_sample_data()
    analyzer = PeakAnalyzer(strategy="auto")
    
    # Find all peaks with minimal filtering
    all_peaks = analyzer.find_peaks(data)
    print(f"All peaks: {len(all_peaks)}")
    
    # Filter by prominence
    prominent_peaks = all_peaks.filter_by(min_prominence=1.0)
    print(f"Prominent peaks (prominence > 1.0): {len(prominent_peaks)}")
    
    # Filter by area
    large_peaks = all_peaks.filter_by(min_area=5)
    print(f"Large peaks (area > 5): {len(large_peaks)}")
    
    # Multiple filters
    filtered_peaks = all_peaks.filter_by(
        min_prominence=0.5,
        min_area=3
    )
    print(f"Filtered peaks (prominence > 0.5, area > 3): {len(filtered_peaks)}")


if __name__ == "__main__":
    # Run basic demonstration
    data, peaks, X, Y = basic_peak_detection()
    
    # Show filtering capabilities
    demonstrate_filtering()
    
    # Plot results if matplotlib is available
    try:
        plot_results(data, peaks, X, Y)
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    
    print("\nExample completed successfully!")
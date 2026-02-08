# Core Algorithm: Topography-Aware Multidimensional Peak Detection

_← [Back to README](../../README.md) | [日本語版](algorithm_ja.md) →_

## The Limitation of 1D Logic in Higher Dimensions

While `scipy.signal.find_peaks` is a robust tool for 1D signal processing, its underlying logic is specialized for one-dimensional topography. In 1D, a peak is simply a point higher than its immediate left and right neighbors. In 2D or N-dimensional spaces, this definition becomes insufficient.

Existing 2D peak detection algorithms frequently encounter two primary issues:

1. **Plateau Misidentification:** Areas of constant height (plateaus) are often incorrectly flagged as multiple individual peaks.
2. **Lack of Feature-Based Filtering:** Most tools lack a mechanism to filter peaks based on geomorphological features such as prominence or isolation, as they treat peaks as isolated points rather than structural components of the data.

**PeakAnalyzer** addresses these issues by redefining a peak not as a "point," but as a **topographic region**.

---

## Topographic Features Extracted

PeakAnalyzer quantifies the geometric properties of peaks to allow for meaningful selection and filtering:

* **Peak Coordinates:** The centroid or representative point of the peak region.
* **Height:** The absolute value at the peak.
* **Prominence:** The vertical distance between the peak and its lowest contour line.
* **Area:** The spatial extent of the peak region.
* **Sharpness:** The local curvature or rate of height change.
* **Isolation:** The distance to the nearest higher peak.
* **Distance:** Spatial relationship metrics between detected features.

### Pixel Scale and Distance Metrics

* **Pixel Scale Parameter:** Allows specification of real-world scale for each dimension (e.g., `scale=[1.0, 0.5]` for different x/y resolution).
* **Minkowski Distance:** Distance calculations support configurable Minkowski distance metrics with parameter `p` (p=1: Manhattan, p=2: Euclidean, p=∞: Chebyshev).
* **Distance Scale:** Physical units can be preserved through scale parameters, ensuring accurate real-world distance measurements.
* **Relative Height (rel_height):** Defines peak width as the area enclosed between prominence_base and peak_height at a relative height ratio.

### Coordinate System Architecture

**Index Space vs. Coordinate Space Separation**

PeakAnalyzer maintains a clear distinction between two spatial representations:

* **Index Space (i,j,...):** Internal data array indexing for computational efficiency
  - Used for: Algorithm processing, memory access, connectivity analysis
  - Format: Integer indices matching data array dimensions
  - Example: `peak_indices = (127, 89)` for 2D array access

* **Coordinate Space (x,y,...):** Real-world physical coordinates for user interaction (when applicable)
  - Used for: User input/output, visualization, spatial analysis, filtering
  - Format: Float coordinates with physical units
  - Example: `peak_coordinates = (-120.5, 36.2)` for longitude/latitude, data value = 1450.0 (arbitrary physical units)

**Physical Meaning and Real-World Units**
* **Distance & Area Calculations:** All measurements computed in actual physical units
* **Anisotropic Resolution Support:** Each axis can have different scales (`scale=[1.0, 1.0, 0.5]`)
* **Unit Preservation:** Maintains consistent physical units throughout analysis pipeline
* **Scale-Aware Features:** Geometric properties respect real-world dimensions

**Spatial Analysis Capabilities**
* **GIS-Compatible Operations:** Coordinate-based filtering, buffering, and spatial queries
* **Geographic Integration:** Native support for geographic coordinate systems
* **Multi-Scale Analysis:** Seamless handling of different resolution data
* **Projection Support:** Flexible coordinate system transformations

**Visualization and User Interface Principles**
* **User-Centric Coordinates:** All user interactions in intuitive xyz... coordinate space
* **Real-World Visualization:** Plots and displays use physical coordinates with proper units
* **Index Abstraction:** Internal ijk... indexing hidden from user interface
* **Intuitive Filtering:** Spatial filters operate on meaningful coordinate ranges

---

## Rigorous Plateau Handling

The core of multidimensional peak detection lies in the correct identification of plateaus. PeakAnalyzer treats a peak as a "region of equal height" to prevent the generation of artifacts within flat surfaces.

### Plateau Detection Procedure

1. **Maximum Filtering:** Apply a local maximum filter to the dataset.
2. **Dilation:** Perform a morphological dilation on the identified regions of constant height .
3. **Validation:** If no cells with height  exist in the dilated boundary that were not part of the original region, the area is classified as a **true peak plateau**.

This logic ensures that internal cells of a large flat top are not misidentified as separate peaks.

---

## Prominence Calculation

PeakAnalyzer adheres strictly to the geomorphological definition of prominence, extending it to multidimensional spaces:

1. A height-priority neighborhood search begins from the peak region.
2. The search continues until it encounters a point with a higher value than the current peak.
3. The lowest elevation reached during this search is defined as prominence_base.
4. The prominence is calculated as: `prominence = peak_height - prominence_base`

### Window Length (wlen) Parameter

The `wlen` parameter constrains prominence calculation to a specified distance:
* When `wlen` is specified, prominence calculation stops at points beyond the given distance from the peak
* This effectively treats the analysis region as a windowed subset of the full dataset
* Useful for focusing on local prominence within specific spatial scales

### Highest Peak Prominence Handling

For peaks that are globally highest (no higher peaks exist):
* **Global Maximum Rule:** Prominence equals the height difference to the lowest point on the data boundary
* **Boundary Condition:** If infinite boundary is used, prominence calculation traces to actual data edges

### Same-Height Connected Peaks with Virtual Peak Creation

When multiple peaks share the same elevation and are topographically connected:
* **Connected Same-Height Peaks at Any Elevation:**
  - When multiple peaks A and B exist at the same height `h` and are connected through terrain of equal or higher elevation
  - **Step 1**: Detect all peaks at the same elevation `h` that are connected via terrain at height ≥ `h`
  - **Step 2**: Find saddle points between connected same-height peaks A and B
  - **Step 3**: Set prominence_base for individual peaks A and B to the height of their connecting saddle point
  - **Step 4**: Create a virtual peak encompassing all connected peaks at height `h`
  - **Step 5**: If higher terrain at height `h' > h` connects to any peak in the virtual peak group via a saddle point, set the virtual peak's prominence_base to that saddle height
  - **Step 6**: The virtual peak's prominence = `h - saddle_height_to_higher_terrain`
  - **Step 7**: If no higher terrain exists, use boundary-based prominence calculation
  - This approach ensures consistent prominence calculation for any elevation level with connected peaks

By implementing this via path-finding logic, the algorithm remains dimension-agnostic.

---

## Calculation Strategies

The library provides two strategies depending on the "topography" of the data:

| Strategy | Logic | Best Use Case |
| --- | --- | --- |
| **Independent Calculation** | Identifies all plateaus first, then calculates prominence for each individually. | Data with high contrast and isolated, sharp peaks. |
| **Batch Heap Search** | Uses a heap-priority queue to explore the terrain by height. | Smooth terrains or data with expansive, complex plateau structures. |

---

## Efficiency via Lazy Evaluation

Calculating all topographic features for every potential peak is computationally expensive. PeakAnalyzer utilizes a **LazyDataFrame** approach to optimize performance.

* **On-Demand Computation:** Features are only calculated when explicitly requested or required for a filtering operation.
* **Dynamic Process:** If a user filters by `prominence > 0.5`, the algorithm calculates prominence only. If a subsequent filter for `area` is added, it then computes the area for the remaining candidates.
* **Memory Efficiency:** This prevents the overhead of maintaining large feature matrices for peaks that are ultimately discarded.

---

## Design Principles

1. **Topographic Focus:** Treat peaks as regions, not dimensionless points.
2. **Strict Plateau Logic:** Ensure flat surfaces are handled without artifact generation.
3. **Geomorphological Accuracy:** Extend 1D prominence definitions to N-dimensions without simplification.
4. **Optimized Search:** Utilize heap-priority structures to maintain logical consistency across dimensions.
5. **Computational Economy:** Use lazy evaluation to minimize unnecessary calculations on large datasets.
6. **Coordinate System Separation:** Maintain clear distinction between index space (ijk...) for internal processing and coordinate space (xyz...) for user interaction.
7. **Physical Meaning Priority:** All user-facing measurements and visualizations use real-world units and coordinates.
8. **Anisotropic Compatibility:** Native support for different resolutions and scales across dimensions.
9. **Spatial Analysis Integration:** Enable GIS-like coordinate-based analysis and filtering capabilities.

---

_For detailed software architecture information, see [Architecture Documentation](../architecture/architecture.md)._

_For implementation specifics, see [Implementation Details](implementation_details.md)._

_For API specifications, see [API Reference](../api/api_reference.md)._
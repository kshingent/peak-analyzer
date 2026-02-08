# Algorithm Implementation Details

_← [Back to Algorithm](algorithm.md) | [Architecture](../architecture/architecture.md) | [API Reference](../api/api_reference.md) | [日本語版](implementation_details_ja.md) →_

## Union-Find Strategy Algorithm

### **Core Challenge**: Height-Priority Processing Without False Peaks

**Step 1: Priority Queue Initialization**
- Create priority queue with all data points as (height, coordinates)
- Use max-heap to process from highest to lowest elevation
- Initialize Union-Find structure for all data points

**Step 2: Height-Level Batch Processing**
- **Critical Insight**: Process ALL points of same height simultaneously
- Extract all points with current maximum height from queue
- Create temporary batch of same-height points for unified processing

**Step 3: Same-Height Connectivity Analysis with Progressive Union**
- **Critical Issue**: Naive union within same-height batch creates false peaks
- **Solution**: Wave-front expansion from already-processed points

**Wave-Front Expansion Algorithm:**
```
processed_points = set()  # From previous height levels
current_batch = get_same_height_points(current_height)
newly_processed = set()  # Points processed in current iteration

# Iterative wave-front expansion
while True:
    temp_store = set()  # Temporarily store newly connected points
    
    # Find unprocessed points connected to processed points
    for point in current_batch:
        if point not in newly_processed:
            for neighbor in get_k_neighbors(point):
                if neighbor in processed_points or neighbor in newly_processed:
                    # Point connects to already processed terrain
                    temp_store.add(point)
                    
                    # Union operations
                    if not has_region(point):
                        # Point-to-region union (first connection)
                        neighbor_region = find_region(neighbor)
                        union_point_to_region(point, neighbor_region)
                    else:
                        # Region-to-region union (subsequent connection)
                        point_region = find_region(point)
                        neighbor_region = find_region(neighbor)
                        if point_region != neighbor_region:
                            union_regions(point_region, neighbor_region)
                    break
    
    # No new connections found - terminate
    if not temp_store:
        break
    
    # Add newly connected points to processed set
    newly_processed.update(temp_store)

# Handle remaining unprocessed points (potential new peaks)
remaining_points = current_batch - newly_processed
isolated_components = find_connected_components(remaining_points, k_connectivity)
for component in isolated_components:
    register_as_new_peak_candidate(component)
```

**Key Benefits:**
- **Prevents False Peaks**: No interior plateau points processed independently
- **Maintains Connectivity**: Proper region merging when multiple processed neighbors exist
- **Wave-Front Processing**: Mimics natural water-flow expansion from higher terrain

**Step 4: Region Validation and Peak Detection**
- **Seed-Connected Regions**: Mark as non-peak (connected to higher terrain)
- **Isolated Regions**: Mark as peak candidates (no connection to higher terrain)
- **Orphaned Points**: Individual points not connected to any seeds → potential new peaks

**Step 4: Union-Find Integration**
- For each plateau component:
  1. Union all points within the component
  2. Check connectivity to previously processed higher plateaus
  3. If connected to higher terrain, mark as non-peak plateau
  4. If isolated from higher terrain, mark as candidate peak plateau

**Step 5: Prominence Calculation During Traversal**
- As we descend through height levels:
  1. Track saddle points for each peak plateau
  2. When a peak connects to higher terrain, record saddle elevation
  3. Calculate prominence as (peak_height - saddle_height)

**Step 6: False Peak Prevention**
- **Key Strategy**: Never process individual points of same-height plateaus separately
- Always process entire same-height connected components as atomic units
- Reject any component that connects to equal-or-higher elevation terrain

### **Queue Management Strategy**
```
while priority_queue not empty:
    current_height = peek_max_height(queue)
    same_height_batch = extract_all_with_height(queue, current_height)
    
    # Process entire batch atomically
    components = find_connected_components(same_height_batch, k_connectivity)
    
    for component in components:
        if is_isolated_from_higher_terrain(component):
            register_as_peak(component)
        else:
            mark_as_non_peak(component)
        
        union_all_points_in_component(component)
```

### **Union-Find Integration Logic**
- **Progressive Union Strategy**: Start from processed points, expand into unprocessed same-height points
- **Two-Phase Union**:
  1. **Point-to-Region**: Unprocessed point joins existing region (first connection)
  2. **Region-to-Region**: Existing regions merge when connected via unprocessed point (subsequent connections)
- **Seed-Based Processing**: Only points connected to higher processed terrain act as integration seeds
- **Prominence Tracking**: Maintain saddle elevation for each region root during expansion

## Plateau-First Strategy Algorithm  

### **Phase 1: Plateau Detection Logic**

**Step 1: Local Maximum Identification**
- For each cell (i,j,...), apply local maximum filter using specified k-connectivity
- Cell is candidate if `data[i,j,...] >= max(all_k_connected_neighbors)`
- This creates binary mask of potential peak cells
- **Issue**: Non-peak plateaus are also detected by this filter

**Step 2: Connected Component Analysis**
- Among candidate cells, group those with identical height values
- Use Union-Find or flood-fill to find connected components using k-connectivity
- Each component represents a potential plateau region of constant height

**Step 3: Plateau Validation (Dilation Test)**
- **Key Insight**: True peak plateaus vs. non-peak plateaus behave differently under dilation
- For each connected component of height `h`:
  1. Create binary mask of the component
  2. Apply morphological dilation using k-connectivity structuring element
  3. Check dilated boundary: `dilated_mask AND NOT original_mask`
  4. **Critical Logic**: If ANY boundary cell has height = `h` (same height), reject as non-peak plateau
  5. If ALL boundary cells have height < `h` (strictly lower), accept as true peak plateau

**Reasoning**: 
- True peak plateaus: Dilation boundary will always be strictly lower
- Non-peak plateaus: Dilation boundary will contain cells of same height (connected to higher terrain). Note that dilation boundary may include both original plateau interior points (missed by local maximum filter due to connection to higher regions) and external points

### **Phase 2: Prominence Calculation**
- For each validated plateau, perform breadth-first search from boundary
- Track minimum elevation until reaching higher terrain
- Calculate prominence as height difference

## Edge/Boundary Handling
- **Infinite Height Boundary**: Pad data edges with maximum float value
- **Infinite Depth Boundary**: Pad data edges with minimum float value  
- **Periodic Boundary**: Wrap data edges with opposite edge values
- **Custom Boundary**: User-specified constant values at edges
- **Artifact Removal**: Filter out peaks too close to data boundaries

## N-Dimensional Connectivity
- **1-Connectivity**: Face sharing (2n neighbors)
- **2-Connectivity**: Face + edge sharing
- **3-Connectivity**: Face + edge + vertex sharing  
- **...**
- **n-Connectivity**: All boundary sharing (3^n-1 neighbors)
- **Efficiency**: Precomputed offset arrays for each connectivity level
- **Custom Patterns**: User-defined neighbor offset patterns
- **Efficiency**: Precomputed offset arrays for fast neighbor generation

## Performance and Memory Considerations
- **Lazy Evaluation**: Features computed only when requested
- **Memory Mapping**: Large arrays handled via memory-mapped files
- **Chunk Processing**: Data divided into overlapping chunks for memory efficiency
- **Parallel Processing**: Multi-threaded feature calculation
- **Caching**: Intelligent caching of expensive computations
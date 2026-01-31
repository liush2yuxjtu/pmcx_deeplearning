# Visual Comparison of MCX Simulation Results

## Maximum Intensity Projection (MIP) Comparison
![MIP Comparison](mip_comparison.png)

## Slice Comparisons
### X-slice
![X-slice Comparison](slice_x_comparison.png)

### Y-slice
![Y-slice Comparison](slice_y_comparison.png)

### Z-slice
![Z-slice Comparison](slice_z_comparison.png)

## Summary
This comparison shows the results from MCX simulation:
1. **Simple Mode**: Using `direct_mcx_run` with numpy space coordinates (real GPU simulation)
2. **Mock Full Mode**: Generated from simple mode results (for comparison purposes only)

Note: Full mode simulation was disabled to save computational resources.
The comparison includes MIPs along all three axes and center slices for each method.

# Visuomotor mismatch responses as a hallmark of explaining away in causal inference

These files accompany the results obtained in *Visuomotor mismatch responses as a hallmark of explaining away in causal inference*.
<!---
 For further details please refer to the manuscript posted on [arxiv](link).
-->

## How to recreate results

Run `julia mismatch.jl` in the folder 'fig1'.

To plot results run `python plot_mismatch.py ../../mismatch/logs/()/log.h5`, where `()` is to be replaced with the desired folder.


## Requirements

The results in this paper were created using `Julia 1.3.1` and `Python 3.6` with `matplotlib`, `numpy` and `h5py`.

### Julia Packages

```julia
pkgs = ["BSON",
"Dates",
"DelimitedFiles",
"FileIO",
"HDF5",
"ImageMagick",
"Images",
"InteractiveUtils",
"JSON",
"LinearAlgebra",
"MAT",
"MLDatasets",
"MultivariateStats",
"Plots",
"Profile",
"ProgressMeter",
"PyPlot",
"SparseArrays"]

using Pkg

Pkg.add(pkgs)
```

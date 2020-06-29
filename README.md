# ColdRice

ML & DL framework using **dask** for its parallel computing, **numpy** for its backend and **numba** for its JIT and CUDA support.

# Setup
* Create the environment from _coldrice.yml_ file if there isn't any: ```conda env create -f coldrice.yml```
* Activate _coldrice_ environment: ```conda activate coldrice```
* Export your active environment to _coldrice.yml_ file: ```conda env export | grep -v "^prefix: " > coldrice.yml```
* Update the environment using the _coldrice.yml_ file: ```conda env update -f coldrice.yml --prune```

# TODOs
* Computational Graph for Gradient Descent
* Add ML Toolkit
* Visualization
* Architecture Search
* CUDA Support

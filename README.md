# ColdRice

ML & DL framework using **dask** for its parallel computing, **numpy** & **jax** for its backend and **numba** for its JIT and CUDA support.

# Setup
* Create the environment with the latest version of python: ```conda create -n coldrice python=3```
* Activate _coldrice_ environment: ```conda activate coldrice```
* Update the environment using the _coldrice.yml_ file: ```conda env update -f coldrice.yml --prune```
* Export your active environment to _coldrice.yml_ file: ```conda env export | grep -v "^prefix: " > coldrice.yml```

###### :warning: You can't create an environment if the environment was exported on a different platform than the target machine.
###### :information_source: `coldrice.yml` was exported on Linux.

# TODOs
* Computational Graph for Gradient Descent
* Add ML Toolkit
* Visualization
* Architecture Search
* CUDA Support

# n2gem

This library provides evaluation metrics, ```density``` & ```coverage```, for generative models based on nearest-neighbor search. The codebase has been developed with ```Python 3.6```, ```Pytorch 1.8.1``` and ```faiss-gpu 1.7.1``` with ```CUDA 10.2```. These metrics have been adopted from ```prdc```(https://github.com/clovaai/generative-evaluation-prdc). The package ```n2gem``` improves upon it by using a faiss index tree for the similarity search. Currently in this version, two options are available namely, `indexflatl2` and `indexivfflat`. If gpu/s is avaiable the computations are automatically performed on them.


## Usage
The package can be used with both faiss-cpu and faiss-gpu. Install ```faiss``` using conda as recommended and follow the instructions given at (https://faiss.ai/). Alternatively, use the ```yml``` files to create the conda environment,
```
conda env create -f <cpu/gpu-env>.yml
```
or create a virtual environment and install the dependencies from ```requirements.txt``` by,
```
python3 -m pip install -r requirements.txt
```
Install the ```n2gem``` package by,
```
python3 -m pip install -e .
```

## Example
A short example on how to use the package. For a detailed note, check the analysis part in the ```ad_attack``` folder.

A faiss index can be given as an input to obtain the metrics,
```python
import torch
import numpy as np
from n2gem.aux_funcs import gem_build_tree
from n2gem.metrics import gem_density, gem_coverage

n_samples = 1024 # no. of samples
n_dims = 12 # no. of dimensions

# create torch tensors
real_embedd = torch.rand((n_samples, n_dims))
gen_embedd = torch.rand((n_samples, n_dims))
nearest_k = 3 # no. of nearest neighbors

# numpy array can also be used
#real_embedd = np.random.normal(loc=0.0, scale=1.0,size=(n_samples, n_dims))
#gen_embedd = np.random.normal(loc=0.0, scale=1.0,size=(n_samples, n_dims))

# build the faiss tree
gem_tree = gem_build_tree(train_samples=real_embedd, nsamples=real_embedd.shape[0], 
                            index_type='indexflatl2')

# compute the metrics, density & coverage
density_value = gem_density(real_tree=gem_tree, real_samples=real_embedd, 
                            gen_samples=gen_embedd, nk=nearest_k)
coverage_value = gem_coverage(real_tree=gem_tree, real_samples=real_embedd, 
                            gen_samples=gen_embedd, nk=nearest_k)

```
else, the build the tree internally by specifying,

```python
from n2gem.metrics import gem_build_density, gem_build_coverage

# compute the metrics directly by creating the faiss tree within
density_value = gem_build_density(real_samples=real_embedd, no_samples=real_embedd.shape[0], 
                                    gen_samples=gen_embedding, index_type='indexflatl2', nk=nearest_k)
coverage_value = gem_build_coverage(real_samples=real_embedd, no_samples=real_embedd.shape[0], 
                                    gen_samples=gen_embedding, index_type='indexflatl2', nk=nearest_k)
```
## Directory Structure
- ```src``` - conatins the files for the ```n2gem``` package
- ```tests``` - contains the test cases
- ```ad_attack``` - this folder contains all the analysis of the metrics and their behaviour for diffrent attacks

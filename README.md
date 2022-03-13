# n2gem

This library provides evaluation metrics, ```density``` & ```coverage```, for generative models based on nearest-neighbor search. The codebase has been developed with ```Python 3.6```, ```Pytorch 1.8.1``` and ```faiss-gpu 1.7.1``` with ```CUDA 10.2```. 


## Usage
### Installation
- pip install n2gem

### Required packages
The package can be used with both faiss-cpu and faiss-gpu. Install ```faiss``` using conda as recommended and follow the instructions given at (https://faiss.ai/). Alternatively, use the ```yml``` files to create the conda environment,
```
conda env create -f <cpu/gpu-env>.yml
```

## Implementation & Example
These metrics have been adopted from ```prdc```(https://github.com/clovaai/generative-evaluation-prdc). The package ```n2gem``` improves upon it by using a faiss index tree for the similarity search. Currently in this version, two options are available namely, `indexflatl2` and `indexivfflat`. If gpu/s is avaiable the computations are automatically performed on them.

A faiss index can be given as an input to obtain the metrics,
```python
import torch
from n2gem.aux_funcs import build_tree_gem
from n2gem.metrics import gem_density, gem_coverage

real_embedd = torch.rand((1024,12), dtype=torch.float32)
gen_embedd = torch.rand((1024,12), dtype=torch.float32)
nearest_k = 3

# build the faiss tree
gem_tree = build_tree_gem(real_embedd, real_embedd.shape[0], 'indexflatl2')

# compute the metrics, density & coverage
density_value = gem_density(gem_tree, real_embedd, gen_embedd, nearest_k)
coverage_value = gem_coverage(gem_tree, real_embedd, gen_embedd, nearest_k)

```
else, the build the tree internally by specifying,

```python
from n2gem.metrics import gem_build_density, gem_build_coverage

# compute the metrics directly by creating the faiss tree within
density_value = gem_build_density(real_embedd, real_embedd.shape[0], gen_embedding, 'indexflatl2', nk=nearest_k)
coverage_value = gem_build_coverage(real_embedd, real_embedd.shape[0], gen_embedding, 'indexflatl2', nk=nearest_k)
```
## Directory Structure
- ```src``` - conatins the files for the ```n2gem``` package
- ```tests``` - contains the test cases
- ```ad_attack``` - this folder contains all the analysis of the metrics and their behaviour for diffrent attacks

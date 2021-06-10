# n2gem

This library provides evaluation metrics for generative models based on nearest-neighbor search. The codebase has been developed with python version 3.6, Pytorch version 1.8.1 and faiss-gpu version 1.7.1 and CUDA 10.2. 

## Required packages
The package can be used with both faiss-cpu and faiss-gpu. Install [faiss] using conda as recommended and follow the instructions given at (https://faiss.ai/). 

## Usage
### Installation
- pip install n2gem

### Example
The metrics such as density and coverage have been implemented from [prdc]. A faiss index is built based on similarity search and 2 index namely, `indexflatl2` and `indexivfflat` is available with this version. If gpu/s is avaiable the computations are automatically performed on them.

A faiss index can be given as an input to obtain the metrics,
```python
import torch
from n2gem.aux_funcs import build_tree_gem
from n2gem.metrics import gem_density, gem_coverage

real_embedd = torch.rand((1024,12), dtype=torch.float32)
gen_embedd = torch.rand((1024,12), dtype=torch.float32)
nearest_k = 3
gem_tree = build_tree_gem(real_embedd, real_embedd.shape[0], 'indexflatl2')

density_value = gem_density(gem_tree, real_embedd, gen_embedd, nearest_k)
coverage_value = gem_coverage(gem_tree, real_embedd, gen_embedd, nearest_k)

```
else, the build the tree internally by specifying,

```python
from n2gem.metrics import gem_build_density, gem_build_coverage

density_value = gem_build_density(real_embedd, real_embedd.shape[0], gen_embedding, 'indexflatl2', nk=nearest_k)
coverage_value = gem_build_coverage(real_embedd, real_embedd.shape[0], gen_embedding, 'indexflatl2', nk=nearest_k)
```

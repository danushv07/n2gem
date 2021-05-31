
import pytest
import faiss
import torch 
import numpy as np
import faiss.contrib.torch_utils
from n2gem.aux_funcs import build_tree_gem
from n2gem.metrics import gem_density, gem_coverage, gem_build_density, gem_build_coverage


d =24
number_of_samples = 10000
xb_real = np.random.random((number_of_samples, d))
xq_gen = np.random.random((1000, d))
#real_samples = torch.from_numpy(xb_real.astype(np.float32))
#ind = build_tree_gem(real_samples, 2000, 'indexivfflat')
gemed_density = gem_build_density(xb_real, 2000, xq_gen)		

import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np
from .aux_funcs import build_tree_gem

def gem_density(real_tree, real_samples, gen_samples, nk=5):
    """ implemetion of  density using faiss tree
    Density counts how many real-sample neighbourhood spheres contain generate samples.
    This implementation uses torch as numerical API

    input parameters:
    real_tree    :: a faiss.IndexFlatL2 or IndexFlatIP tree
                    that was filled with all real samples
    real_samples :: numpy array of N real samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32
    gen_samples  :: numpy array of N generated samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32

    output parameters:
    density :: single float within [0, inf)
    """
    # convert tensors 
    real_samples = torch.from_numpy(real_samples.astype(np.float32))
    gen_samples = torch.from_numpy(gen_samples.astype(np.float32))
    
    real_fake_dists = torch.cdist(real_samples, gen_samples)

    #do a +1 as faiss query results always include the query object itself
    #at distance 0
    D, _ = real_tree.search(real_samples, nk+1)
    real_maxradii, _ = torch.max(torch.sqrt(D), dim=1)

    assert real_maxradii.shape[0] == real_samples.shape[0], f"""
    shapes don't match {real_maxradii} vs {real_samples.shape}"""

    density_mask = (1. / float(nk)) * (
            real_fake_dists <
            real_maxradii.reshape(*real_maxradii.shape, 1)
    )

    print("[density_]", density_mask.dtype)

    value = density_mask.sum(dim=0).mean()

    return value



def gem_coverage(real_tree, real_samples, gen_samples, nk=5):
    """ implemeting of coverage using faiss tree

    Coverage measures the fraction of real samples whose
    neighbourhoods contain at least one fake sample.

    input parameters:
    real_tree    :: a faiss.IndexFlatL2 or IndexFlatIP tree
                    that was filled with all real samples

    real_samples :: numpy array of N real samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32
    gen_samples  :: numpy array of N generated samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32

    output parameters:
    coverage :: single float within [0, 1]
    """
    
    # convert tensors 
    real_samples = torch.from_numpy(real_samples.astype(np.float32))
    gen_samples = torch.from_numpy(gen_samples.astype(np.float32))
    
    real_fake_dists = torch.cdist(real_samples, gen_samples)

    #do a +1 as faiss includes the the query object itself
    D, _ = real_tree.search(real_samples, nk+1)
    real_maxradii, _ = torch.max(torch.sqrt(D), dim=1)

    assert real_maxradii.shape[0] == real_samples.shape[0]

    real_fake_mins, _ = real_fake_dists.min(dim=1)

    coverage_mask = (
            real_fake_mins < real_maxradii
    )
    value = coverage_mask.to(dtype=torch.float32).mean()

    return value


def gem_build_density(real_samples, no_samples, gen_samples, index_type, n_cells=100, nk=5):
    """ implemetation of density 
    - creates a faiss index tree followed by density computation
    
    Density counts how many real-sample neighbourhood spheres contain generate samples.
    This implementation uses torch as numerical API

    input parameters:
    -----------------
    real_samples : TYPE - numpy array of N real samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
                    
    no_samples : nsamples : TYPE - integer
               DESCRIPTION - number of samples to be considered for building the tree
               
    gen_samples  : TYPE - numpy array of N generated samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
    
    index_type : TYPE - string
                 DESCRIPTION - the index type to choose from faiss ['indexflatl2', 'indexivfflat']
                                default - 'indexflatl2'
    
    n_cells : TYPE - integer
              DESCRIPTION - number of voronoi cells
                              default = 100
    nk : TYPE - integer
         DESCRIPTION - the value for the nearest neighbours
         
    output parameters:
    -------------------
    density :: single float within [0, inf)
    """
    
    # convert from numpy to torch tensors
    real_samples = torch.from_numpy(real_samples.astype(np.float32))
    gen_samples = torch.from_numpy(gen_samples.astype(np.float32))
    
    # build the tree
    real_tree = build_tree_gem(real_samples, no_samples, index_type, n_cells)
    
    real_fake_dists = torch.cdist(real_samples, gen_samples)

    #do a +1 as faiss query results always include the query object itself
    #at distance 0
    D, _ = real_tree.search(real_samples, nk+1)
    real_maxradii, _ = torch.max(torch.sqrt(D), dim=1)

    assert real_maxradii.shape[0] == real_samples.shape[0], f"""
    shapes don't match {real_maxradii} vs {real_samples.shape}"""

    density_mask = (1. / float(nk)) * (
            real_fake_dists <
            real_maxradii.reshape(*real_maxradii.shape, 1)
    )

    #print("[density_]", density_mask.dtype)

    value = density_mask.sum(dim=0).mean()
    print(value)
    return value


def gem_build_coverage(real_samples, no_samples, gen_samples, index_type, n_cells=100, nk=5):
    """ implemetation of coverage 
    - creates a faiss index tree followed by density computation
    
    Coverage measures the fraction of real samples whose
    neighbourhoods contain at least one fake sample.
    This implementation uses torch as numerical API

    input parameters:
    -----------------
    real_samples : TYPE - numpy array of N real samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
                    
    no_samples : nsamples : TYPE - integer
               DESCRIPTION - number of samples to be considered for building the tree
               
    gen_samples  : TYPE - numpy array of N generated samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
    
    index_type : TYPE - string
                 DESCRIPTION - the index type to choose from faiss ['indexflatl2', 'indexivfflat']
                                default - 'indexflatl2'
    
    n_cells : TYPE - integer
              DESCRIPTION - number of voronoi cells
                              default = 100
    nk : TYPE - integer
         DESCRIPTION - the value for the nearest neighbours
         
    output parameters:
    coverage :: single float within [0, 1]
    """
    
    # convert from numpy to torch tensors
    real_samples = torch.from_numpy(real_samples.astype(np.float32))
    gen_samples = torch.from_numpy(gen_samples.astype(np.float32))
    
    # build the tree
    real_tree = build_tree_gem(real_samples, no_samples, index_type, n_cells)
    
    real_fake_dists = torch.cdist(real_samples, gen_samples)

    #do a +1 as faiss query results always include the query object itself
    #at distance 0
    D, _ = real_tree.search(real_samples, nk+1)
    real_maxradii, _ = torch.max(torch.sqrt(D), dim=1)
    real_fake_mins, _ = real_fake_dists.min(dim=1)

    assert real_maxradii.shape[0] == real_samples.shape[0], f"""
    shapes don't match {real_maxradii} vs {real_samples.shape}"""

    coverage_mask = (
            real_fake_mins < real_maxradii
    )
    value = coverage_mask.to(dtype=torch.float32).mean()

    return value
   

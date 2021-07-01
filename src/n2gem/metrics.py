import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np
from .aux_funcs import gem_build_tree

# the device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#GLOBAL_DEVICE = torch.device(DEVICE_STRING)


def input_conv(real, gen):
    """
    Function to check and convert the real and gen samples to torch.Tensor
    
    Parameters
    ----------------
    real : TYPE - numpy or Torch.Tensor
           size: N real samples of dimensionality D
    gen : TYPE - numpy or Torch.Tensor
           size: N generated samples of dimensionality D
           
    Return
    ----------------
    real : TYPE - Torch.Tensor of dtype:float32
           size: N real samples of dimensionality D
    gen : TYPE - Torch.Tensor of dtype:float32
           size: N generated samples of dimensionality D
    """
    if isinstance(real, torch.Tensor) and isinstance(gen, torch.Tensor):
        if real.is_cuda:
            real.cpu().to(device)
        else:
            real.to(device)
        if gen.is_cuda:
            gen.cpu().to(device)
        else:
            gen.to(device)
            
        if not (real.dtype == torch.float32) or (gen.dtype == torch.float32): 
            real = real.to(torch.float32)
            gen = gen.to(torch.float32)
        
    elif isinstance(real, np.ndarray) and isinstance(gen, np.ndarray):
        real = torch.from_numpy(real.astype(np.float32)).to(device)
        gen = torch.from_numpy(gen.astype(np.float32))
        
    else:
        print("Both the given inputs should be either numpy array or torch.Tensor")
    
    return real, gen


def gem_density(real_tree, real_samples, gen_samples, nk=5):
    """ implemetion of  density using faiss tree
    Density counts how many real-sample neighbourhood spheres contain generate samples.
    This implementation uses torch as numerical API

    input parameters:
    real_tree    :: a faiss.IndexFlatL2 or IndexFlatIP tree
                    that was filled with all real samples
    real_samples :: numpy array or Torch.Tensor of N real samples of dimensionality D,
                    shape should be (N,D)
    gen_samples  :: numpy array  or Torch.Tensor of N generated samples of dimensionality D,
                    shape should be (N,D)

    output parameters:
    density :: single float within [0, inf)
    """
    # convert to torch tensors 
    real_samples, gen_samples = input_conv(real_samples, gen_samples)
    
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

    value = density_mask.sum(dim=0).mean()

    return value


def gem_coverage(real_tree, real_samples, gen_samples, nk=5):
    """ implemeting of coverage using faiss tree

    Coverage measures the fraction of real samples whose
    neighbourhoods contain at least one fake sample.

    input parameters:
    real_tree    :: a faiss.IndexFlatL2 or IndexFlatIP tree
                    that was filled with all real samples

    real_samples :: numpy array or torch.Tensor of N real samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32
    gen_samples  :: numpy array or torch.Tensor of N generated samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32

    output parameters:
    coverage :: single float within [0, 1]
    """
    
    # convert to torch tensors 
    real_samples, gen_samples = input_conv(real_samples, gen_samples)
    
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


def gem_build_density(real_samples, no_samples, gen_samples, index_type, n_cells=100, probe=100, 
                      nk=5, verbose=0):
    """ implemetation of density 
    - creates a faiss index tree followed by density computation
    
    Density counts how many real-sample neighbourhood spheres contain generated samples.
    This implementation uses torch as numerical API

    input parameters:
    -----------------
    real_samples : TYPE - numpy array or Torch.tensor of N real samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
                    
    no_samples : TYPE - integer
               DESCRIPTION - number of samples to be considered for building the tree
               
    gen_samples  : TYPE - numpy array or torch.Tensor of N generated samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
    
    index_type : TYPE - string
                 DESCRIPTION - the index type to choose from faiss ['indexflatl2', 'indexivfflat']
                                default - 'indexflatl2'
    
    n_cells : TYPE - integer
              DESCRIPTION - number of voronoi cells
              default = 100
                              
    probe : TYPE - integer
            DESCRIPTION - number of voronoi cells to be visited
            default = 100 
            
    nk : TYPE - integer
         DESCRIPTION - the value for the nearest neighbours
         
    verbose : TYPE - integer
           DESCRIPTION - the option to print statements 
                         default = 0 - print no statements
                         1 - print information regarding dataset
                         2 - print the type of tree and time taken
         
    output parameters:
    -------------------
    density :: single float within [0, inf)
    """
    
    # convert to torch tensors 
    real_samples, gen_samples = input_conv(real_samples, gen_samples)
    
    # build the tree
    real_tree = gem_build_tree(real_samples, no_samples, index_type, n_cells, probe, verbose=verbose)
    
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
    return value


def gem_build_coverage(real_samples, no_samples, gen_samples, index_type, n_cells=100, probe=100, 
                       nk=5, verbose=0):
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
              
    probe : TYPE - integer
            DESCRIPTION - number of voronoi cells to be visited
            default = 100
            
    nk : TYPE - integer
         DESCRIPTION - the value for the nearest neighbours
         
    verbose : TYPE - integer
           DESCRIPTION - the option to print statements 
                         default = 0 - print no statements
                         1 - print information regarding dataset
                         2 - print the type of tree and time taken
         
    output parameters:
    coverage :: single float within [0, 1]
    """
    
    # convert to torch tensors 
    real_samples, gen_samples = input_conv(real_samples, gen_samples)
    
    # build the tree
    real_tree = gem_build_tree(real_samples, no_samples, index_type, n_cells, probe, verbose=verbose)
    
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
   
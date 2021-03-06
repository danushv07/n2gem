import numpy as np
#import h5py as h5
import torch
import faiss
import faiss.contrib.torch_utils 
import time

# the device to use
DEVICE_STRING = 'cuda' if torch.cuda.is_available() else 'cpu'
GLOBAL_DEVICE = torch.device(DEVICE_STRING)


def build_index(index, ngpus):
    """
    function to build an index on the gpu
    
    Parameters
    ------------------
    index : TYPE - faiss index of the type indexxflatl2 or indxivflat
    
    ngpus : TYPE - integer, indicating the number of gpus
    
    Return
    ---------------
    ind_tree : TYPE - faiss index built using gpu
    """
    gpu_resource = faiss.StandardGpuResources() # declare a gpu memory
    if not ngpus > 1:
        ind_tree = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
    else:
        ind_tree = faiss.index_cpu_to_all_gpus(index)
    
    return ind_tree


def index_add(index, dataset, n_samples, index_type, verbose, n_cels=100, pncells=100):
    """
    function to add dataset to the tree
    
    Parameters
    --------------
    index : TYPE - faiss index of the type indexxflatl2 or indxivflat
    
    dataset : TYPE - TYPE - torch tensor of N real samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
    
    nsamples : TYPE - integer
               DESCRIPTION - number of samples to be considered for building the tree
    
    index_type : TYPE - string
                 DESCRIPTION - the index type to choose from faiss ['indexflatl2', 'indexivfflat']
                                default - 'indexflatl2'
    
    n_cels : TYPE - integer
             DESCRIPTION - number of voronoi cells
             default=100                 
            
    pncells :  TYPE - integer
               DESCRIPTION - number of voronoi cells to be visited
               default=100               
    
   Return
   ---------------
   index - faiss tree of given index_type with the dataset added to the tree
   
    """
    
    s_time = time.time()
    index.add(dataset[:n_samples, :])
    e_time = time.time() - s_time
    
    if not index_type == 'indexflatl2':
        index.nprobe = pncells
        if verbose==2:
            print(f"Creating the tree by {index_type} using {n_cels} cells took {e_time:.5f} sec.")
    else:
        if verbose==2:
            print(f"Creating the tree by {index_type} took {e_time:.5f} sec")
    
    return index

                
def gem_build_tree(train_samples, nsamples, index_type='indexflatl2', n_cells=100, p_cells=100, seed=42,
                  verbose=0):
    """
    Build a faiss tree
    
    Parameters
    -----------------
    train_samples : TYPE - torch tensor of N real samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
                    
    nsamples : TYPE - integer
               DESCRIPTION - number of samples to be considered for building the tree
    
    index_type : TYPE - string
                 DESCRIPTION - the index type to choose from faiss ['indexflatl2', 'indexivfflat']
                                default - 'indexflatl2'
    
    n_cells : TYPE - integer
              DESCRIPTION - number of voronoi cells
                              default = 100
    p_cells : TYPE - integer
              DESCRIPTION - number of voronoi cells to be visited
                              default = 100
                              
    seed : TYPE - integer
           DESCRIPTION - the seed for the generator
                         default = 42
    
    verbose : TYPE - integer
           DESCRIPTION - the option to print statements 
                         default = 0 - print no statements
                         1 - print information regarding dataset
                         2 - print the type of tree and time taken
    
    Return
    ------------------
    tree : TYPE - faiss tree of the specified index type
    
    """
    np.random.seed(seed)
    
    # get & print the no. of gpus
    ngpus = faiss.get_num_gpus()
    
    if (verbose==1) or (verbose==2):
        print(f"The tree is created on the dataset of shape {train_samples.shape} using {GLOBAL_DEVICE}")
    
    # check for the given input train_samples
    # check only if used independently
    if isinstance(train_samples, torch.Tensor):
        if not (train_samples.dtype == torch.float32): 
            train_samples = train_samples.to(torch.float32)
            
    elif isinstance(train_samples, np.ndarray):
        if not (train_samples.dtype == np.float32): 
            train_samples = train_samples.astype(np.float32)
            
    else:
        print("The given input is neither numpy array or torch.Tensor")
    
    dataset = train_samples    
    total_samples = dataset.shape[0] if (nsamples < 0 or nsamples > dataset.shape[0]) else nsamples
    
    # build the tree
    index = faiss.IndexFlatL2(dataset.shape[-1])
    #gpu_resource = faiss.StandardGpuResources() # declare a gpu memory
    
    
    if ('cuda' in DEVICE_STRING):
        
        # indexflatl2 in gpu
        if (index_type == 'indexflatl2'):
            index_tree = build_index(index, ngpus)
            return index_add(index_tree, dataset, total_samples, index_type, verbose)
            
        # indexivfflat in gpu    
        elif (index_type == 'indexivfflat'):
            flat_index = faiss.IndexIVFFlat(index, dataset.shape[-1], n_cells)
            flat_ind_gpu = build_index(flat_index, ngpus)
            flat_ind_gpu.train(dataset)
            return index_add(flat_ind_gpu, dataset, total_samples, index_type, verbose, n_cells, p_cells)
            
        else:
            print(f"Other index type not available")
      
    
    elif ('cpu' in DEVICE_STRING):
        
        # indexflatl2 in cpu
        if (index_type == 'indexflatl2'):
            return index_add(index, dataset, total_samples, index_type, verbose)
        
        # indexivflat in cpu
        elif (index_type == 'indexivfflat'):
            flat_index = faiss.IndexIVFFlat(index, dataset.shape[-1], n_cells)
            flat_index.train(dataset)
            return index_add(flat_index, dataset, total_samples, index_type, verbose, n_cells, p_cells)
        
        else:
            print(f"Other index type not available")

        

import numpy as np
import h5py as h5
import torch
import faiss
import faiss.contrib.torch_utils # needed to build tree using torch tensors
import time


# the device to use
DEVICE_STRING = 'cuda' if torch.cuda.is_available() else 'cpu'
GLOBAL_DEVICE = torch.device(DEVICE_STRING)


def build_tree_gem(train_samples, nsamples, index_type='indexflatl2', n_cells=100, seed=42):
    """
    Build a faiss tree
    
    Parameters
    -----------------
    train_samples : TYPE - numpy array of N real samples of dimensionality D,
                    DESCRIPTION - shape should be (N,D)
                    
    nsamples : TYPE - integer
               DESCRIPTION - number of samples to be considered for building the tree
    
    index_type : TYPE - string
                 DESCRIPTION - the index type to choose from faiss ['indexflatl2', 'indexivfflat']
                                default - 'indexflatl2'
    
    n_cells : TYPE - integer
              DESCRIPTION - number of voronoi cells
                              default = 100
    seed : TYPE - integer
           DESCRIPTION - the seed for the generator
                         default = 42
    
    Return
    ------------------
    tree : TYPE - faiss tree of the specified index type
    
    """
    np.random.seed(seed)
    
    # get & print the no. of gpus
    ngpus = faiss.get_num_gpus()
    #print("number of gpus: ", ngpus)
    
    print(f"The tree is created on the dataset of shape {train_samples.shape} using {GLOBAL_DEVICE}")
    
    #dataset = train_samples.to(GLOBAL_DEVICE)
    dataset = train_samples
    total_samples = dataset.shape[0] if (nsamples < 0 or nsamples > dataset.shape[0]) else nsamples
    
    # build the tree
    index = faiss.IndexFlatL2(dataset.shape[-1])
    gpu_resource = faiss.StandardGpuResources() # declare a gpu memory
    
    
    if ('cuda' in DEVICE_STRING):
        # indexflatl2 in gpu
        if (index_type == 'indexflatl2'):
            if not ngpus > 1:
                index_tree = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
            else:
                index_tree = faiss.index_cpu_to_all_gpus(index)
                

            start_time = time.time()
            index_tree.add(dataset[:total_samples, :])
            delta_time = time.time() - start_time
            print(f"Creating the tree by {index_type} took {delta_time:.5f} sec")

            return index_tree
            
        # indexivfflat in gpu    
        elif (index_type == 'indexivfflat'):
            flat_index = faiss.IndexIVFFlat(index, dataset.shape[-1], n_cells)
            if not ngpus > 1:
                flat_ind_gpu = faiss.index_cpu_to_gpu(gpu_resource, 0, flat_index)
            else:
                flat_ind_gpu = faiss.index_cpu_to_all_gpus(flat_index)
            
            flat_ind_gpu.train(dataset)
            start_time = time.time()
            flat_ind_gpu.add(dataset[:total_samples, :])
            delta_time = time.time() - start_time
            print(f"Creating the tree by {index_type} using {n_cells} cells took {delta_time:.5f} sec.") 

            return flat_ind_gpu
            
        else:
            print(f"Other index type not available")
      
    
    elif ('cpu' in DEVICE_STRING):
        if (index_type == 'indexflatl2'):
            start_time = time.time()
            index.add(dataset[:total_samples, :])
            delta_time = time.time() - start_time
            print(f"Creating the tree by {index_type} took {delta_time:.5f} sec")
            return index
        
        elif (index_type == 'indexivfflat'):
            flat_index = faiss.IndexIVFFlat(index, dataset.shape[-1], n_cells)
            flat_index.train(dataset)

            start_time = time.time()
            flat_index.add(dataset[:total_samples, :])
            delta_time = time.time() - start_time
            print(f"Creating the tree by {index_type} using {n_cells} cells took {delta_time:.5f} sec.") 

            return flat_index
        else:
            print(f"Other index type not available")

    else:
        print(f"Other index types not available")
        

import pytest
import faiss
import torch 
import numpy as np
import faiss.contrib.torch_utils
from n2gem.aux_funcs import build_tree_gem
from n2gem.metrics import gem_density, gem_coverage, gem_build_density, gem_build_coverage
from sklearn.datasets import make_blobs

#d =24
#number_of_samples = 10000
#xb_real = np.random.random((number_of_samples, d))
#xq_gen = np.random.random((1000, d))
#real_samples = torch.from_numpy(xb_real.astype(np.float32))
#ind = build_tree_gem(real_samples, 2000, 'indexivfflat')
#gemed_density = gem_build_density(xb_real, 2000, xq_gen)		

def BlobDataset():
    """
    Test dataset created using sklearn make_blobs
    
    real - numpy array of 1024 samples and 2 dimensions
    gen - numpy array of 256 samples and 2 dimensions
    
    """
    real, _ =  make_blobs(n_samples=1024, n_features=2, centers=[(0,0), (5,5)], random_state=42)
    gen, _ = make_blobs(n_samples=256, n_features=2, centers=[(1,1)], random_state=42)
    
    return real.astype(np.float32), gen.astype(np.float32)


def test_given_BlobDataset_return_dimesion():
    """
    Test case for the dimensions of the dataset returned from BlobDataset
    
    """
    real, gen = BlobDataset()
    assert real.shape == (1024, 2)
    assert gen.shape == (256, 2)
 

def test_given_BlobDataset_build_tree_gem_with_IndexFlatl2_return_index_distance():
    """
    Test case for build_tree_gem with IndexFlatl2
    
    Expected results:
    ------------------
    - the type of index tree should be IndexFlatl2
    - the type of Distances should be numpy.ndarray
    - the 1st Id should be 1023
    - the 1st Distance value should be 0
    
    
    """
    real, gen = BlobDataset()
    index_tree = build_tree_gem(real, real.shape[0])
    D, I = index_tree.search(real[-1:, ...], 5)
    
    assert isinstance(index_tree, faiss.IndexFlatL2)
    assert isinstance(D, np.ndarray)
    
    assert I[0,0] == 1023
    assert D[0,0] == 0
 
def test_given_BlobDataset_build_tree_gem_with_IndexIVFFlat_return_index_distance():
    """
    Test case for build_tree_gem with indexIVFFlat
    
    Expected results:
    ------------------
    - the type of index tree should be IndexIVFFlat
    - the type of Distances should be numpy.ndarray
    - the 1st Id should be 1023
    - the 1st Distance value should be 0
    
    
    """
    real, gen = BlobDataset()
    index_tree = build_tree_gem(real,real.shape[0], 'indexivfflat')
    D, I = index_tree.search(real[-1:, ...], 5)
    
    assert not isinstance(index_tree, faiss.IndexFlatL2)
    assert isinstance(D, np.ndarray)
    
    assert I[0,0] == 1023
    assert D[0,0] == 0
   

def test_given_BlobDataset_build_tree_gem_with_IndexFlatl2_return_torch_distance():
    """
    Test case for build_tree_gem with IndexFlatl2
    
    Expected results:
    ------------------
    - the type of index tree should be IndexFlatl2
    - the type of Distances should be torch.Tensor
    - the 1st Id should be 1023
    - the 1st Distance value should be 0
    
    
    """
    real, gen = BlobDataset()
    index_tree = build_tree_gem(real, real.shape[0], 'indexflatl2')
    
    realn = torch.from_numpy(real).detach().requires_grad_(False)
    
    D, I = index_tree.search(realn[-1:, ...], 5)
    
    assert isinstance(index_tree, faiss.IndexFlatL2)
    assert not isinstance(D, np.ndarray)
    
    assert I[0,0] == 1023
    assert D[0,0] == 0
 

def test_given_BlobDataset_build_tree_gem_with_IndexFlatl2_compare_distance_IndexFlatL2():
    """
    Test case for build_tree_gem with IndexFlatl2
    
    Expected results:
    ------------------
    - the type of index tree should be IndexFlatl2
    - the distance should be same
    """
    real, gen = BlobDataset()
    index_tree = build_tree_gem(real, real.shape[0], 'indexflatl2')
    D, _ = index_tree.search(real[-1:, ...], 5)
    
    index_test_tree = faiss.IndexFlatL2(real.shape[-1])
    index_test_tree.add(real[:real.shape[0], :])
    D_ip, _ = index_test_tree.search(real[-1:, ...], 5)
    
    assert isinstance(index_tree, faiss.IndexFlatL2)
    assert np.array_equal(D, D_ip)

    
def test_given_BlobDataset_build_tree_gem_with_IndexFlatl2_compare_distance_IndexIVFFlat():
    """
    Test case for build_tree_gem with IndexFlatl2
    
    Expected results:
    ------------------
    - the type of index tree should be IndexIVFFlat
    - the distance should be same
    """
    real, gen = BlobDataset()
    index_tree = build_tree_gem(real,real.shape[0], 'indexivfflat', n_cells=2)
    D, _ = index_tree.search(real[-1:, ...], 5)
    print("D: ", D)

    inddy = faiss.IndexFlatL2(real.shape[1])
    index_test_tree = faiss.IndexIVFFlat(inddy, real.shape[-1], 2)
    index_test_tree.train(real)
    index_test_tree.add(real)
    D_ip, _ = index_test_tree.search(real[-1:, ...], 5)
    print("D_ip: ", D)
    assert isinstance(index_tree, faiss.IndexIVFFlat)
    assert np.array_equal(D, D_ip)
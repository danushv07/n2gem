
import pytest
import faiss
import torch 
import numpy as np
import faiss.contrib.torch_utils
from n2gem.aux_funcs import build_tree_gem
from n2gem.metrics import gem_density, gem_coverage, gem_build_density, gem_build_coverage
from sklearn.datasets import make_blobs
import prdc

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
    real, _ =  make_blobs(n_samples=1024, n_features=24, centers=[(0,0), (5,5)], random_state=42)
    gen, _ = make_blobs(n_samples=256, n_features=24, centers=[(1,1)], random_state=42)
    
    return real.astype(np.float32), gen.astype(np.float32)


def CreateTree(real, indtype):
    
    real_tree = faiss.IndexFlatL2(real.shape[-1])
    if indtype =='indexflatl2':
        real_tree.add(real)
        return real_tree
    
    elif indtype == 'indexivfflat':
        ivftree = faiss.IndexIVFFlat(real_tree, real.shape[-1], 2)
        ivftree.train(real)
        ivftree.add(real)
        return ivftree
        
        
def test_given_BlobDataset_return_dimesion():
    """
    Test case for the dimensions of the dataset returned from BlobDataset
    
    """
    real, gen = BlobDataset()
    assert real.shape == (1024, 12)
    assert gen.shape == (256, 12)
 

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
    
    index_test_tree = faiss.IndexFlatL2(real.shape[1])
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
    

    inddy = faiss.IndexFlatL2(real.shape[1])
    index_test_tree = faiss.IndexIVFFlat(inddy, real.shape[-1], 2)
    index_test_tree.train(real)
    index_test_tree.add(real)
    D_p, _ = index_test_tree.search(real[-1:, ...], 5)
    
    assert isinstance(index_tree, faiss.IndexIVFFlat)
    assert np.array_equal(D, D_p)

    
def test_given_BlobDataset_real_tree_return_density_from_gem_density():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexflatl2')
    
    real_gem_density = gem_density(test_tree, real, gen, nk=5)

    real = torch.from_numpy(real)
    gen = torch.from_numpy(gen)
    nk = 5

    real_fake_dists = torch.cdist(real, gen)
    D, _ = test_tree.search(real, nk+1)
    real_maxradii,_ = torch.max(torch.sqrt(D), axis=1)
    density_mask = (1. / float(nk)) * (
            real_fake_dists <
            real_maxradii.reshape(*real_maxradii.shape, 1)
    )

    expected_density_test_value = density_mask.sum(dim=0).mean()

    assert(real_gem_density == expected_density_test_value)
    assert isinstance (real_gem_density, torch.Tensor)
    assert isinstance (expected_density_test_value, torch.Tensor)
    

def test_given_BlobDataset_real_samples_indexflatl2_return_density_from_gem_build_density():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexflatl2')
    
    real_gem_density = gem_build_density(real, real.shape[0], gen, 'indexflatl2')

    real = torch.from_numpy(real)
    gen = torch.from_numpy(gen)
    nk = 5

    real_fake_dists = torch.cdist(real, gen)
    D, _ = test_tree.search(real, nk+1)
    real_maxradii,_ = torch.max(torch.sqrt(D), axis=1)
    density_mask = (1. / float(nk)) * (
            real_fake_dists <
            real_maxradii.reshape(*real_maxradii.shape, 1)
    )

    expected_density_test_value = density_mask.sum(dim=0).mean()

    assert(real_gem_density == expected_density_test_value)
    assert isinstance (real_gem_density, torch.Tensor)
    assert isinstance (expected_density_test_value, torch.Tensor)

    
def test_given_BlobDataset_real_samples_indexivfflat_return_density_from_gem_build_density():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexivfflat')
    
    real_gem_density = gem_build_density(real, real.shape[0], gen, 'indexivfflat', n_cells=2)

    real = torch.from_numpy(real)
    gen = torch.from_numpy(gen)
    nk = 5

    real_fake_dists = torch.cdist(real, gen)
    D, _ = test_tree.search(real, nk+1)
    real_maxradii,_ = torch.max(torch.sqrt(D), axis=1)
    density_mask = (1. / float(nk)) * (
            real_fake_dists <
            real_maxradii.reshape(*real_maxradii.shape, 1)
    )

    expected_density_test_value = density_mask.sum(dim=0).mean()

    assert(real_gem_density == expected_density_test_value)
    assert isinstance (real_gem_density, torch.Tensor)
    assert isinstance (expected_density_test_value, torch.Tensor)
 

def test_given_BlobDataset_real_tree_return_coverage_from_gem_coverage():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexflatl2')
    
    real_coverage_density = gem_coverage(test_tree, real, gen, nk=5)

    real = torch.from_numpy(real)
    gen = torch.from_numpy(gen)
    nk = 5

    real_fake_dists = torch.cdist(real, gen)
    D, _ = test_tree.search(real, nk+1)
    real_maxradii,_ = torch.max(torch.sqrt(D), axis=1)
    real_fake_mins, _ = real_fake_dists.min(dim=1)

    coverage_mask = (
            real_fake_mins < real_maxradii
    )
    expected_coverage_test_value = coverage_mask.to(dtype=torch.float32).mean()
    

    assert(real_coverage_density == expected_coverage_test_value)
    assert isinstance (real_coverage_density, torch.Tensor)
    assert isinstance (expected_coverage_test_value, torch.Tensor)

    
def test_given_BlobDataset_real_samples_indexflatl2_return_coverage_from_gem_build_coverage():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexflatl2')
    
    real_gem_coverage = gem_build_coverage(real, real.shape[0], gen, 'indexflatl2')

    real = torch.from_numpy(real)
    gen = torch.from_numpy(gen)
    nk = 5

    real_fake_dists = torch.cdist(real, gen)
    D, _ = test_tree.search(real, nk+1)
    real_maxradii,_ = torch.max(torch.sqrt(D), axis=1)
    real_fake_mins, _ = real_fake_dists.min(dim=1)

    coverage_mask = (
            real_fake_mins < real_maxradii
    )
    expected_coverage_test_value = coverage_mask.to(dtype=torch.float32).mean()

    assert(real_gem_coverage == expected_coverage_test_value)
    assert isinstance (real_gem_coverage, torch.Tensor)
    assert isinstance (expected_coverage_test_value, torch.Tensor)
    
    
def test_given_BlobDataset_real_samples_indexivfflat_return_coverage_from_gem_build_coverage():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexivfflat')
    
    real_gem_coverage = gem_build_coverage(real, real.shape[0], gen, 'indexivfflat', n_cells=2)

    real = torch.from_numpy(real)
    gen = torch.from_numpy(gen)
    nk = 5

    real_fake_dists = torch.cdist(real, gen)
    D, _ = test_tree.search(real, nk+1)
    real_maxradii,_ = torch.max(torch.sqrt(D), axis=1)
    real_fake_mins, _ = real_fake_dists.min(dim=1)

    coverage_mask = (
            real_fake_mins < real_maxradii
    )
    expected_coverage_test_value = coverage_mask.to(dtype=torch.float32).mean()

    assert(real_gem_coverage == expected_coverage_test_value)
    assert isinstance (real_gem_coverage, torch.Tensor)
    assert isinstance (expected_coverage_test_value, torch.Tensor)

    
def test_given_BlobDataset_compare_prdc_density_gem_density():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexflatl2')
    real_gem_density = gem_density(test_tree, real,  gen)
    
    ref = prdc.compute_prdc(real, gen, 5)
    
    assert np.allclose(real_gem_density, ref['density'])    

    
def test_given_BlobDataset_compare_prdc_density_gem_build_density():
    
    real, gen = BlobDataset()
    real_gem_density = gem_build_density(real, real.shape[0], gen, 'indexflatl2')
    
    ref = prdc.compute_prdc(real, gen, 5)
    
    assert np.allclose(real_gem_density, ref['density'])

    
def test_given_BlobDataset_compare_prdc_coverage_gem_coverage():
    
    real, gen = BlobDataset()
    test_tree = CreateTree(real, 'indexflatl2')
    real_gem_coverage = gem_coverage(test_tree, real,  gen)
    
    ref = prdc.compute_prdc(real, gen, 5)
    
    assert np.allclose(real_gem_coverage, ref['coverage'])    

    
def test_given_BlobDataset_compare_prdc_coverage_gem_build_coverage():
    
    real, gen = BlobDataset()
    real_gem_coverage = gem_build_coverage(real, real.shape[0], gen, 'indexflatl2')
    
    ref = prdc.compute_prdc(real, gen, 5)
    
    assert np.allclose(real_gem_coverage, ref['coverage'])
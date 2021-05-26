import faiss

import faiss.contrib.torch_utils
import torch

import numpy as np
from scipy.spatial.distance import cdist

"""
Reliable Fidelity and Diversity Metrics for Generative Models (ICML 2020,
https://arxiv.org/abs/2002.09797) implemented using faiss similarity search
"""


def density(real_tree, real_samples, gen_samples, nk=5):
    """ implemeting density from the prcd paper, 2002.09797
    Density counts how many real-sample neighbourhood spheres contain generate samples.

    input parameters:
    real_tree    :: a faiss.IndexFlatL2 or IndexFlatIP tree
                    that was filled with all real samples

    real_samples :: numpy array of N real samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32
    gen_samples  :: numpy array of N generated samples of dimensionality D,
                    shape should be (N,D); data type must be np.float32

    output parameters: density :: single float within [0, inf)
    """

    real_fake_dists = cdist(real_samples, gen_samples)

    #do a +1 as faiss query results always include the query object itself
    #at distance 0
    D, _ = real_tree.search(real_samples, nk+1)

    #maximum values, shape (real_samples.shape[0],)
    real_maxradii = np.max(np.sqrt(D), axis=1)

    assert real_maxradii.shape[0] == real_samples.shape[0]

    density = (1. / float(nk)) * (
            real_fake_dists <
            np.expand_dims(real_maxradii, axis=1)
    ).sum(axis=0).mean()

    return density

def density_(real_tree, real_samples, gen_samples, nk=5):
    """ implemeting density from the prcd paper, 2002.09797
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


def coverage(real_tree, real_samples, gen_samples, nk=5):
    """ implemeting coverage from the prcd paper

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

    real_fake_dists = cdist(real_samples, gen_samples)

    #do a +1 as faiss includes the the query object itself
    D, _ = real_tree.search(real_samples, nk+1)
    real_maxradii = np.max(np.sqrt(D), axis=1)

    assert real_maxradii.shape[0] == real_samples.shape[0]

    coverage = (
            real_fake_dists.min(axis=1) <
            real_maxradii
    ).mean()

    return coverage

def coverage_(real_tree, real_samples, gen_samples, nk=5):
    """ implemeting coverage from the prcd paper

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


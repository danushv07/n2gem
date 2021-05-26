import numpy as np
import h5py as h5
import torch
import faiss
import faiss.contrib.torch_utils # needed to build tree using torch tensors

def build_tree(indextype,
          # ntrees,
          device,

          inputfile,
          inputname,
          nsamples,

          output,
          verbose,
          seed
          ):
    """ train a flat faiss index on 1D data """

    np.random.seed(seed)
    be_verbose = verbose > 0
    #device_to_use, n_cpus = interpret_device(device)
    #if n_cpus != None and n_cpus > 0:
    #    torch.set_num_threads(n_cpus)
    #    click.echo(f">> running in {device_to_use}, with {torch.get_num_threads()} cores")
    #else:
    #    click.echo(f">> running in {device_to_use}")

    ############################################################################
    #
    # Loading the data
    #
    h5f = h5.File(inputfile, "r")

    try:
        h5ds = h5f[inputname] if inputname in h5f.keys() else None
    except:
        click.echo(f"{inputname} not found in {inputfile}. Exiting.")
        return 1

    assert h5ds, f'{inputname} not extracted from {inputfile}'

    h5ds_shape = h5ds.shape

    assert len(h5ds.shape) == 2, f"dataset is not 1D, {h5ds.shape}"
    totalsamples = h5ds_shape[0] if (nsamples < 0 or nsamples > h5ds_shape[0]) else nsamples


    h5dst = torch.from_numpy(np.asarray(h5ds).astype(np.float32)).to(device_to_use)

    ###############################################################
    # train tree
    index = faiss.IndexFlatL2(h5dst.shape[-1])
    if not "indexflatl2" in indextype.lower():
        click.echo(">> other index types no implemented yet, exiting")
        return 1

    click.echo(f">> adding {h5dst.shape} {h5dst.dtype}s ({totalsamples} samples)")
    start = time.time()
    index.add(h5dst[:totalsamples, ...])
    deltat = time.time() - start

    click.echo(">> creating tree took {deltat:.4f} sec".format(deltat=deltat))

    faiss.write_index(index, str(output))
    click.echo(f">> faiss {indextype} written to {output}")





# the device to use
DEVICE_STRING = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GLOBAL_DEVICE = torch.device(DEVICE_STRING)


def build_tree_gem(file, input_name, output_file, nsamples, seed=42):
    """
    Build a faiss tree
    
    Parameters
    -----------------
    file : TYPE - string
           DESCRIPTION - the path to hdf5 file containing the train/real samples to build tree
                          size:(n_samples, n_dimensions)
             
    input_name : TYPE - string
                 DESCRIPTION - name of the variable in the file
    
    output_file : TYPE - string
                  DESCRIPTION - the path to store the tree(s)
    
    nsamples : TYPE - integer
                    DESCRIPTION - number of samples to be considered for building the tree
    
    seed : TYPE - integer
           DESCRIPTION - the seed for the generator
                         default = 42
    
    
    """
    np.random.seed(seed)
    
    # get & print the no. of gpus
    ngpus = faiss.get_num_gpus()
    #print("number of gpus: ", ngpus)
    
    print(f">> running in {GLOBAL_DEVICE}")
    input_file = h5.File(file, "r")
    
    try:
        data_for_tree = input_file[input_name] if input_name in input_file.keys() else None
    except:
        print(f"{input_name} not found in {file}. Exiting.")
        return 1
    
    dataset = torch.from_numpy(np.asarray(data_for_tree).astype(np.float32)).to(GLOBAL_DEVICE)
    #print(dataset.shape)
    
    total_samples = dataset.shape[0] if (nsamples < 0 or nsamples > dataset.shape[0]) else nsamples
    
    # build the tree
    index = faiss.IndexFlatL2(dataset.shape[-1])
    gpu_resource = faiss.StandardGpuResources() # declare a gpu memory
    
    if 'cuda' in DEVICE_STRING:
        if not ngpus > 1:
            index_tree = faiss.index_cpu_to_gpu(gpu_resource, 0, index) # single gpu
        else:
            index_tree = faiss.index_cpu_to_all_gpus(index) # multiple gpus
        
        index_tree.add(dataset[:total_samples, :])
    
    else:
        index.add(dataset[:total_samples, :])
        
    #faiss.write_index(index, str(output_file))

# import libraries
import numpy as np
import h5py as h5
import torch

import faiss



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





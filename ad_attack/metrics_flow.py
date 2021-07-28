# dependencies
import os 
os.environ["GIT_PYTHON_REFRESH"] = "quiet" 
import foolbox as fb
import torch
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import numpy as np
from sklearn.model_selection import train_test_split
from n2gem.metrics import gem_build_coverage, gem_build_density
import click

@click.command()
@click.option('mdi','--model_images_path', default='images/model_dataset_images.pt',
             help='the path to load the entire model_dataset images')
@click.option('mdl','--model_labels_path', default='images/model_dataset_labels.pt',
             help='the path to load the entire model_dataset labels')
@click.option('vi','--vali_images_path', default='images/validation_images.pt',
             help='the path to load the entire validation_dataset images')
@click.option('vl','--vali_lables_path', default='images/validation_labels.pt',
             help='the path to load the entire validation_dataset labels')
@click.option('ms','--model_batch', default=0.3, type=float, 
             help='the size of model_dataset to be chosen for metrics computation')
@click.option('vs','--vali_batch_size', default=100, type=int,
             help='the size of each batch of validation_images')
@click.option('e','--epsilon', default=None,
             help='the epsilon value to be considered')
def batch_metrics(model_images_path, model_labels_path, vali_images_path, vali_lables_path,
                 model_batch, vali_batch_size, epsilon):
    """
    Function to compute density & coverage between model_dataset & validation_dataset in batches
    of specified size
    
    """
    
    # read the model_dataset & validation dataset
    md_images = torch.load(model_images_path, map_location='cpu').cpu()
    md_labels = torch.load(model_labels_path, map_location='cpu').cpu()

    vali_images = torch.load(vali_images_path, map_location='cpu')
    vali_labels = torch.load(vali_lables_path, map_location='cpu')
    
    # create the model_dataset for the metric (either entire dataset or batch of it using stratify split)
    if model_batch is None:
        model_images = md_images
    else:
        _, model_images, _, _ = train_test_split(md_images.numpy(), md_labels.numpy(), test_size=model_batch, random_state=42, 
                                                 stratify=md_labels.numpy())
        model_images = torch.from_numpy(np.array(md_images))
    
    # convert to torch tensor
    realx = model_images.view(model_images.shape[0], -1).to(device)
    vali_images = torch.from_numpy(np.array(vali_images)).to(device)
    vali_labels = torch.from_numpy(np.array(vali_labels)).to(device)
    
    # form the batches for the validation dataset
    batches = np.arange(0, len(vali_images), vali_batch_size)
    vali_images = vali_images.view(vali_images.shape[0], -1)
    batch_model_metrics = []
    
    if epsilon is None:
        for i in range(len(batches) - 1):
            start = batches[i]; end= batches[i+1]
            gen_validatex = vali_images[start:end,...]
            gen_labelsx = vali_labels[start:end]
            density_validatex = gem_build_density(real, real.shape[0], gen_validatex, 'indexflatl2')
            coverage_validatex = gem_build_coverage(real, real.shape[0], gen_validatex, 'indexflatl2')
            batch_model_metrics.append([density_validatex.cpu().numpy(), coverage_validatex.cpu().numpy()])
    else:
       epi = np.linspace(0.0, 1, num=20) 
    
    
    batch_model_metrics = np.array(batch_model_metrics)
    np.savetxt((attack_name+'_model_vali_metrics.dat'), batch_model_metrics)
   
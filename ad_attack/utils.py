"""This file contains the helper functions for the notebook 
    adversarial_attck
    Author:Danush Kumar Venkatesh
    Created on 29/11/2021
"""

# install dependencies
import torch
import torchvision
from fastai.vision.all import *
from fastai.vision import *
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import numpy as np
from n2gem.metrics import gem_build_coverage, gem_build_density
import eagerpy as ep
import matplotlib.pyplot as plt



class LoadMed(torch.utils.data.Dataset):
    """
    Class to load and form tensor datasets for the MedMNIST datasets
    
    overload of the torch.utils.data.Dataset
    """
    
    def __init__(self, imgs, labels, transform):
        
        self.images = imgs
        self.labels = labels
        self.transform = transform
        
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        
        img, target = self.images[idx], self.labels[idx].astype(int)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        

        return img, target
    
    
def create_dataset(dataset, vali_split, test_split, train_set=None, test_set=None):
    """Function to split the datasets
    
    1. stratify split to form model_Dataset & validation_set
    2. training and test split based on test_split size
        
        
        Parameters
        ----------------------
        dataset: torch.utils.Dataset(contains both train and test set)
        vali_split: the size for the validation set 
        test_split: the size of the test set
        train_set: (optional) torch.utils.dataset
        test_set: (optional) torch.utils.dataset
        
        Return
        ----------------------
        X_trainset: list, size: train_size, 1, 28, 28
        X_testset: list, size: test_size, 1, 28, 28
        X_validation: list, size: vali_size, 1, 28, 28
        y_trainset: list, size: train_size
        y_testset: list, size: test_size
        y_valiset: list, size: vali_size
        
    """
    # combine the datasets
    if not dataset:
        dataset = ConcatDataset([train_set, test_set])
    
    X=[]; Y=[];
    for i in range(len(dataset)):
        x, y = dataset[i]
        X.append(x.numpy())  #.detach().cpu().numpy())
        Y.append(y)
    
    # split the dataset into train and validation set
    X_model, X_validation, y_model, y_validation = train_test_split(X, Y, test_size=vali_split, random_state=42, stratify=np.array(Y))

    print("X_validation_set size: ", np.array(X_validation).shape)
    print("Y_validation_set size: ", len(y_validation))
    
    # split the X_model, y_model into training set & test set
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X_model, y_model, test_size=test_split, random_state=42, stratify=np.array(y_model))
    print("X_train_set size: ", np.array(X_trainset).shape)
    print("Y_train_set size: ", len(y_trainset))
    print("X_test_set size: ", np.array(X_testset).shape)
    print("Y_test_set size: ", len(y_testset))
    
    #X = np.array(X); Y = np.array(Y)
    
    return X_trainset, X_testset, X_validation, y_trainset, y_testset, y_validation

# convert datasets to torch tensor datasets
def convert_tensor(X_trainset, X_testset, y_trainset, y_testset, X_validation, y_validation):
    """
    Function to convert the split datasets into torch Tensor dataset
    
    Parameters
    ----------------------
    X_trainset: list, size: train_size, 1, 28, 28
    X_testset: list, size: test_size, 1, 28, 28
    X_validation: list, size: vali_size, 1, 28, 28
    y_trainset: list, size: train_size
    y_testset: list, size: test_size
    y_valiset: list, size: vali_size
    
    Return
    -------------------
    train_set: TensorDataset, consists of training images and labels
    test_set: TensorDataset, consists of test images and labels
    validation_set: TensorDataset, consists of validation images and labels
    
    # for the adversarial attack
    model_dataset_images: Tensor, size: (training + test) size, only the images
    model_dataset_images: Tensor, size: (training + test) size, only the labels
    """
    X_trainset = torch.Tensor(X_trainset); 
    y_trainset = torch.Tensor(y_trainset).type(torch.LongTensor).view(-1) #dtype = torch.long);
    X_testset = torch.Tensor(X_testset); 
    y_testset = torch.Tensor(y_testset).type(torch.LongTensor).view(-1);
    X_validation = torch.Tensor(X_validation); 
    y_validation = torch.Tensor(y_validation).type(torch.LongTensor).view(-1)
    
    #print(X_trainset.shape, y_trainset.shape)
    # y - label should be longTensor for fastai training
    
    # form model_dataset
    model_dataset_images = torch.vstack((X_trainset, X_testset))
    model_dataset_labels = torch.cat((y_trainset, y_testset))
    #model_dataset_labels = torch.vstack((y_trainset, y_testset))
    
    # form the Tensor dataset
    train_set = torch.utils.data.TensorDataset(X_trainset, y_trainset)
    test_set = torch.utils.data.TensorDataset(X_testset, y_testset)
    validation_set = torch.utils.data.TensorDataset(X_validation, y_validation)
    
    return train_set, test_set,  model_dataset_images, model_dataset_labels , validation_set


def model_attack(attk, model, images, labels, epsilon):
    """
    Function to perform the adv attack on the pytorch model
    
    Parameters
    --------------------
    attk: type of foolbox attack
    model: foolbox Pytorch modelss
    image: the images used for the attack, either as ep.tensor or torch.tensor
            size:(no_of_samples x dims)
    labels: corresponding labels for the images, either as ep.tensor or torch.tensor
            size:(no_of_samples 
    epsilon: the pertubation
    
    Return
    -------------
    clip_adv: list, the actual adversarial examples generated for the given model
                size: similar to the input images
    adv_ : boolean list, indicating whether a given image is adversarial example or not,
            True - adversarial sample
            False - not an adversarial sample
    """
    raw_adv, clip_adv, adv_ = attk(model, images, labels, epsilons=epsilon)
    
    return clip_adv, adv_


def model_metrics(adv_imgs, real_imgs, epi_len):
    """
    Function to compute density & coverage between the 
    real & adversarial samples
    
    Parameters
    ---------------
    adv_imgs: ep.tensor, the generated adversarial samples, size: no_of_images x image_dims
    real_imgs: torch.tensor, size: no_of_images x dims
    epi_len: integer, the length or number of epsilons used for the attack
    """
    density = []
    coverage = []
    if not epi_len==1:
        for i in range(epi_len):
            # generated adversarial for each epsilon(convert from eagerpy --> torch and reshape)
            gen = adv_imgs[i].raw.view(adv_imgs[i].shape[0], -1).cpu()

            # density
            density.append(gem_build_density(real_imgs, real_imgs.shape[0], gen, 'indexflatl2'))

            # coverage
            coverage.append(gem_build_coverage(real_imgs, real_imgs.shape[0], gen, 'indexflatl2'))
    else:
        gen = adv_imgs.raw.view(adv_imgs.shape[0], -1).cpu()

        # density
        density.append(gem_build_density(real_imgs, real_imgs.shape[0], gen, 'indexflatl2'))

        # coverage
        coverage.append(gem_build_coverage(real_imgs, real_imgs.shape[0], gen, 'indexflatl2'))
        
    return density, coverage


def feature_extractor(image_set, model, f_name, f_needed=True):
    """
    Function for feature extraction of the given Model
    --> the model should be tweeked in amanner that only feature extraction
        operation takes place, i.e., only inout to last layer is taken(avoid softmax)
        
    Parameters
    --------------------
    image_set: the dataset of model or validation images
                either: torch.Tensor or np.array, size:no.of images x dims
    model: the respective Model with the loaded pretrained weights
            check: the device on which model is placed
    f_name: str, the filename to save the generated features in the form of numpy array
            value stored size: no.of_images x output_dim of the layer in the model
    f_needed: bool, default=True
            return the feature generated
    
    """
    features = []
    
    if not isinstance(image_set[0], torch.Tensor):
        image_set = torch.from_numpy(np.stack(image_set))
    
    for i in range(len(image_set)):
        img = image_set[i].unsqueeze(0).to('cpu')
        with torch.no_grad():
            feature = model(img)
        features.append(feature.cpu().detach().numpy().reshape(-1))
    
    features = np.array(features)
    
    if f_name:
        np.savetxt(f_name, features)   
    
    if f_needed: return features

    
def plot_hist(true_label, pred_label, nclass, f_path, nr=4, nc=5, f_zise=(12,10)):
    """
    Function to plot the histogram of true labels vs predicted label
    customized to plot
    Parameters
    --------------
    true_label: list, the true label for the dataset, 
                size: no.of samples
    
    pred_label: list, the predicted label for each epsilon value
                size: len(epsilon) x no.of samples
    
    nclass: int, the number of classes in the dataset
    
    f_path: str, the file path to save the figure
    nr: int, the number of rows in the subplot, default=4
    nc: int, the number of rows in the subplot, default=5
    f_size: tuple, the size of the plot, default=(12,10)
    
    Return
    -------------
    the histogram plot saved in f_path as .png file
    """
    fig, ax = plt.subplots(nr, nc, figsize=f_zise)
    k = 0
    a = [16,17,18,19,20]
    episo = np.arange(0, 1.05, 0.05)
    episo = np.delete(episo, 18)
    ticks = np.arange(0,nclass,1).tolist()
    for i in range(4):
        for j in range(5):

            ax[i,j].hist(np.array(tru_label).ravel(), bins=np.arange(0,nclass+1,1), align='left', alpha=0.5, color='r', edgecolor='k', 
                         label='true_label')
            ax[i,j].hist(pred_label[k], bins=np.arange(0,nclass+1,1), align='left', alpha=0.3, color='b', label='adv_label')
            ax[i,j].set_title((r'$\epsilon$' + f"={episo[k]:.2f}"))
            k+= 1
            if(j!=0):
                ax[i,j].set_yticks([]) 
            if(k not in a):
                ax[i,j].set_xticks([]) 
            else:
                ax[i,j].set_xticks(ticks)
            if(k==a[-1]):
                ax[i,j].legend(loc=(0.5, 0.7), fontsize='small')    
    fig.text(0.5, 0, 'class', va='center', rotation='horizontal', fontsize='large')
    plt.tight_layout()
    #plt.savefig("trial.png", bbox_inches = 'tight',pad_inches = 0)
    plt.savefig((f_path+'.png'), bbox_inches = 'tight')  
    
    
### Metrics for FGSM Attack - stats

#- ```Batch_modelvali_metrics``` - density & coverage between model_dataset(20000) & validation_dataset(2100) in batches of 100 samples
#- ```Batch_modelvali_adv_metrics``` - attack the model in batches using validation_dataset. Computed density & coverage between model_dataset & validation_adv samples

def model_validation_metrics_batches(real, images, labels, batch_size):
    """
    Function to obtain density and coverage between model_dataset & 
    validation_dataset in batches (100 images)
    
    Parameters
    ---------------
    real: the model_dataset of (20000 images) using stratify split, torch.Tensor
    images: the validation_dataset fo (2100 images) , torch.Tensor
    labels: the labels for the validation_dataset, torch.Tensor
    batch_size: int, size of batches of validation images
    
    Return
    -----------------
    bacth_model_vali_metrics: list containing density & coverage
    """
    n_vali_images = images.shape[0]
    batches = np.arange(0, n_vali_images, batch_size)
    

    bacth_model_vali_metrics = []
    images = images.view(images.shape[0], -1)
    for i in range(len(batches)-1):

        # model_dataset & validation_dataset in batches
        start = batches[i]; end= batches[i+1]
        gen_validatex = images[start:end,...]
        gen_labelsx = labels[start:end]
        density_validatex = gem_build_density(real, real.shape[0], gen_validatex, 'indexflatl2')
        coverage_validatex = gem_build_coverage(real, real.shape[0], gen_validatex, 'indexflatl2')
        bacth_model_vali_metrics.append([density_validatex.cpu().numpy(), coverage_validatex.cpu().numpy()])
   
    return bacth_model_vali_metrics

def batch_validation_attack(real, images, labels, epi, batch_size):
    """
    Function to obtain the density & coverage between model_dataset & validation_adv dataset
    
    - attack the model in batches of validation dataset(100 images)
    - for each epsilon value, compute the density & coverage for model_dataset & adv batch
    
    Parameters
    -------------
    real: real: the model_dataset of (20000 images) using stratify split, torch.Tensor
    images: images: the validation_dataset fo (2100 images) , torch.Tensor
    labels: the labels for the validation_dataset, torch.Tensor
    epi: np.array, epsilons for the attack
    batch_size: int, size of batches of validation images
    
    Return
    -------------
    batch_model_vali_adv_metrics: list containing density & coverage
    """
    advs = []
    batch_model_vali_adv_metrics = []
    n_vali_images = images.shape[0]
    batches = np.arange(0, n_vali_images, batch_size)
    images = ep.astensor(images)
    labels = ep.astensor(labels)
    for i in range(len(batches)-1):
        start = batches[i]; end= batches[i+1]
        imagesx = images[start:end,...]
        labelsx = labels[start:end]
        adv_fgsm_batch, adv_fgsm_batch_info = model_attack(attack2, original_model, imagesx, labelsx, epi)
        if(i%5==0):print(f"batch{i}!!!")
        
        for i in range(len(adv_fgsm_batch)):
            gen = adv_fgsm_batch[i].raw.view(len(adv_fgsm_batch[i]), -1)
            densityx = gem_build_density(real, real.shape[0], gen, 'indexflatl2')
            coveragex = gem_build_coverage(real, real.shape[0], gen, 'indexflatl2')
            batch_model_vali_adv_metrics.append([densityx.cpu().numpy(), coveragex.cpu().numpy()])
    
    return batch_model_vali_adv_metrics

def dimPlot(images, labels, path):
    """
    Function to plot the embedding space using pymde
    -> perserve_neighbours option is currently implemented and active
    -> by default the embedding_dim is 2
    
    Parameters
    ---------------
    images: torch.Tensor, the images to be plotted , size:no._of_images x dims
    labels: torch.Tensor, corresponding labels to the images, size:no_of_images
    path: str, the location to store image
    
    Return
    ---------------
    the embedded image is stored in the given path
    """
    embdd = pymde.preserve_neighbors(images, embedding_dim=2, constraint=pymde.Standardized()).embed() 
    #embdd = pymde.preserve_distances(images, embedding_dim=2, constraint=pymde.Standardized()).embed() 
    pymde.plot(embdd, color_by=labels, marker_size=5, background_color='grey', savepath=path)
    
# create the plots for boundary attack
def read_data(f1, sk):
    """
    function to read the given file
    
    Parameters
    -----------
    f1: str, path of the file to be read
    sk: int, no. of rows to be skipped
    """
    df = pd.read_csv(f1, header=None, sep=" ", skiprows=sk)
    return df.iloc[:,1:].values

def computation(c):
    """
    function to compute quantiles and stats on a given matrix
    
    Parameters
    -----------
    c: np.array, no. of batches x samples
    """
    quant = np.quantile(c, [0.25,0.50,0.75], axis=0)
    stats = np.median(c, axis=0)
    return quant, stats

def ins(arr, idx, val):
    return np.insert(arr, 4, val, axis=0)

def bdy_process(f1, prefix=None, f2=None, dataset=False):
    """
    Function to read the file and return the quantile and stats for the
    reference and adv. metrics for the boundary attack plots
    
    Parameters
    ----------------
    f1: str, the file path before the {batch no} prefix
    prefix: int, the batch no or size
    f2: str, the file path after the prefix
    dataset: bool, to combine f1+f2
    
    Return
    ---------------
    den_q, adv_den_q, den_s, adv_den_s: refrence & adv. density for all the batches
    cov_q, adv_cov_q, cov_s, adv_cov_s: refrence & adv. coverage for all the batches
    """
    bmodel_validation_den_q = [];bmodel_validation_adv_den_q=[];bmodel_validation_cov_q=[];bmodel_validation_adv_cov_q=[];
    bmodel_validation_den_s=[];bmodel_validation_adv_den_s=[];bmodel_validation_cov_s=[];bmodel_validation_adv_cov_s=[];
    if(dataset):
        data = read_data(f1, 5)
        d_quant, d_stats = computation(data)
        return d_quant[:,0], d_quant[:,1], d_stats[0], d_stats[1], d_quant[:,2], d_quant[:,3], d_stats[2], d_stats[3]                  
    else:
        for i in prefix:
            data = read_data((f1+str(i)+f2), 7)
            d_quant, d_stats = computation(data)
            # quantile info
            bmodel_validation_den_q.append(d_quant[:,0])
            bmodel_validation_adv_den_q.append(d_quant[:,1])
            bmodel_validation_cov_q.append(d_quant[:,2])
            bmodel_validation_adv_cov_q.append(d_quant[:,3])

            # stats info
            bmodel_validation_den_s.append(d_stats[0])
            bmodel_validation_adv_den_s.append(d_stats[1])
            bmodel_validation_cov_s.append(d_stats[2])
            bmodel_validation_adv_cov_s.append(d_stats[3])

        den_q = np.array(bmodel_validation_den_q)
        adv_den_q = np.array(bmodel_validation_adv_den_q)
        cov_q = np.array(bmodel_validation_cov_q)
        adv_cov_q = np.array(bmodel_validation_adv_cov_q)
        den_s = np.array(bmodel_validation_den_s)
        adv_den_s = np.array(bmodel_validation_adv_den_s)
        cov_s = np.array(bmodel_validation_cov_s)
        adv_cov_s = np.array(bmodel_validation_adv_cov_s)
    
    
        return den_q, adv_den_q, den_s, adv_den_s, cov_q, adv_cov_q, cov_s, adv_cov_s
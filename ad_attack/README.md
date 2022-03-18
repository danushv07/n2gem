This folder contains the code and related files to the experiments conducted to evaluate the metrics. The models were trained using [fastai](https://github.com/fastai/fastai)```version 2.4```, the boimedical images were from [MedMNIST](https://github.com/MedMNIST/MedMNIST)```version 2.0.1```. The attacks were performed using [foolbox](https://github.com/bethgelab/foolbox)```version 3.3.1``` and [matplotlib](https://matplotlib.org/) ```version 3.3.2``` and [sklearn](https://scikit-learn.org/stable/install.html) ```version 0.23.2``` were used for auxiliary measures.

## Directory structure

### Notebooks
- ```adversarial_attck_``` - this notebook contains the procedure to perform both FGSM and Boundary attack to all the given datasets. It also includes the computation of reference and adversarial metrics.
- ```analysis``` - the additional analysis provided in the supplementary material is available in this notebook. Pymde visualizations and computations for the uncertainity bands are added here.
- ```plot_book``` - contains the procedure to reproduce the plots from the paper

### Files
- ```utils.py``` - the auxiliary/helper functions for the notebooks are available here
- ```util_models.py``` - the model architectures used in this study are provided in this file.

### Folders
- ```chkpt_files``` - contains the pretrained weights for each dataset considered in this study
- ```metrics_files``` - to reproduce the plots/analysis performed in this study, the resulst of the experiments and their associated values are stored in ```.dat``` files in this folder. All the files are accessible using ```pandas```.

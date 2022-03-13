# This folder consists of files with the metrics information of the model after the attack

## ```FGSM attack```
- ```FGSM attack``` -> 20 epsilons
### folders
```mnist``` 
    - the metrics files for the MNIST dataset
    - FGSM_attack_density/coverage -> contains the reference & adversarial values after the attack
    - FGSM_attack_42batch -> contains the metrics values for batches of images to construct the uncertainity bands
    - FGSM_attack_100batch_adv -> contains the adversarial metrics values for batches of images to construct the uncertainity bands
    - FGSM_attack_FX -> indicates the metrics value for the feature extracted from the last layer of the model

```mnist_mix```
    - the metrics value for admixture of adv. samples
    - consider only files with randmix 

```organmnist```
    - the metrics files for the OrganMNIST dataset
    - FGSM_attack_density/coverage -> contains the reference & adversarial values after the attack
    - FGSM_attack_100batch -> contains the reference metrics values for batches of images to construct the uncertainity bands
    - FGSM_attack_100batch_adv -> contains the adversarial metrics values for batches of images to construct the uncertainity bands
    - FGSM_attack_FX -> indicates the metrics value for the feature extracted from the last layer of the model
    - ```mix```
        - the metrics value for admixture of adv. samples

```pathmnist```
    - the metrics files for the PathMNIST dataset
    - FGSM_attack_NewNet_density/coverage -> contains the reference & adversarial values after the attack
    - FGSM_attack_100batch -> contains the reference metrics values for batches of images to construct the uncertainity bands
    - FGSM_attack_100batch_adv -> contains the adversarial metrics values for batches of images to construct the uncertainity bands
    - FGSM_attack_FX -> indicates the metrics value for the feature extracted from the last layer of the model
    - ```mix```
        - - the metrics value for admixture of adv. samples
    

## ```Boundary Attack```
### files
```mnist```
    - Boundary_attack_metrics/coverage -> contains reference & adversarial metrics for batch size of 100 images
    - Boundary_attack_<batch_size>batch_metrics -> contains the reference & adversarial metrics for specified batch size
    
```pathmnist```
    - Boundary_attack_pathmnist_<batch_size>batch_metrics -> contains the reference & adversarial metrics for specified batch size
    
```organmnist```
    - Boundary_attack_organmnist_<batch_size>batch_metrics -> contains the reference & adversarial metrics for specified batch size

### fgsm - resnet
- ```FGSM_attack_resnet18_coverage/density``` => attack the resnet model(18,34,50) with validation images of 2100 & metrics between model dataset(22000 images) & validation adversarial images

### fgsm - model_capacity
- ```FGSM_attack_overfit/underfit_density/coverage``` - attack the overfit/underfit(see model_arch) with validation images of 2100 & metrics between model dataset(22000 images) & validation adversarial images


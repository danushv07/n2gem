# This folder consists of files with the metrics information of the model after the attack

## ```FGSM attack```
- attack model with model_dataset -> model_dataset & adversarial samples
- attack model validataion_dataset -> model_dataset & vali_adv samples
- model_dataset & vali_dataset
- ```FGSM attack``` -> 20 epsilons

## ```Boundary Attack```
- file: Boundary_attack_metrics.dat
- density & coverage: 
	- model_dataset(67000) & validation_dataset(2100)
	- model_dataset(67000) & validation_adv(2100)

- density & coverage:
	- model_dataset(67000) & validation_dataset(batches of 100 images)
	- model_dataset(67000) & validation_adv(batches)

## Files
For the FGSM attack, the model_dataset=> 22000 images(startify split), validation_images=> 2100 samples
- ```FGSM_attack_batch_adv_metrics``` => attack the model using batches of validation images of size 100 and compute density & coverage
- ```FGSM_attack_batch_metrics``` => coverage & density between model_dataset & validation dataset in batches of 100 images
- ```FGSM_attack_density``` => model_dataset & validation_set density and model_dataset & adversarial validation_set density
- ```FGSM_attack_coverage``` => model_dataset & validation_set coverage and model_dataset & adversarial validation_set coverage
- ```FGSM_attack_42batch_metrics``` => density & coverage between model_dataset & validation_set in batches of 42 images
- ```FGSM_attack_42batch_adv_metrics``` => density & coverage between model_dataset & adversarial validation_set in batches of 42 images"

### fgsm - resnet
- ```FGSM_attack_resnet18_coverage/density``` => attack the resnet model(18,34,50) with validation images of 2100 & metrics between model dataset(22000 images) & validation adversarial images

### fgsm - model_capacity
- ```FGSM_attack_overfit/underfit_density/coverage``` - attack the overfit/underfit(see model_arch) with validation images of 2100 & metrics between model dataset(22000 images) & validation adversarial images

### fgsm - pathmnist
- ```FGSM_attack_pathmnist_NewNet_density/coverage``` - attack the NewNet(see model_arch) with validation images of 5359 & metrics between model dataset(101821 images) & validation adversarial images
- ```FGSM_attack_pathmnist_NewNet_30model_density/coverage``` - attack the NewNet(see model_arch) with validation images of 5359 & metrics between model dataset(30547 images) & validation adversarial images
- ```FGSM_attack_pathmnist_100batch_30model_metrics``` - coverage & density between model_dataset & validation dataset in batches of 100 images
- ```FGSM_attack_pathmnist_100batch_30model_adv_metrics``` - coverage & density between model_dataset & adv validation dataset in batches of 100 images
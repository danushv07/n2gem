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

# The folder contains files pertaining to various models and their respective checkpoint

- 100Batch_Vali_Adv_LSBU.pt file: the adverasarial samples obtained by using LinearSearchBlendedUniformNoiseAttack
with directions=2000, steps=5000 on the validation_dataset of 2100 samples in batches of 100. This is used as the starting_points for the Boundary attack.


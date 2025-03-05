# (Mis)Fitting Scaling Laws: A Survey of Scaling Laws

This repository contains the data and code to reproduce the figures and tables in the paper ["(Mis)Fitting Scaling Laws: A Survey of Scaling Laws"](https://arxiv.org/abs/2502.18969), by Margaret Li, Sneha Kudugunta and Luke Zettlemoyer.

## Training models

Model training code was hard-forked from [open_lm](https://github.com/mlfoundations/open_lm) and modified. 

We've compiled all of the commands needed to download and preprocess data and to train models in one script:

```
bash scaling_laws/open_lm/misfitting/run_train_scale.sh
```

If you need to make modifications, refer to the commands within the script.
There are strings in this repo which need to be replaced with constants specific to your environment -- e.g., wandb username, slurm account and partition, or conda environment, if relevant. These can be found by grepping for `## TODO: FILL IN`.

Configurations for all of the model architectures we use are listed in `scaling_laws/open_lm/model_configs` and named with the convention `misfitting_{size}`. Additional hyperparameters and training settings may be found in `open_lm/constants/slurm_constants.py`.

## Scaling law fitting and plotting

All of the code used to fit scaling laws and generate plots found in the paper are in `paper_analysis_and_plots.py`. Parts of this code were adapted from the code released by [Besiroglu, et. al.](https://github.com/epoch-research/analyzing-chinchilla) and by [Porian, et. al.](https://github.com/formll/resolving-scaling-law-discrepancies).

## Data

All of the data used to conduct the analyses in our paper can be found in the `data/` folder. This includes data taken from [Besiroglu, et. al.](https://arxiv.org/abs/2404.10102) and [Porian, et. al.](https://arxiv.org/abs/2406.19146), which we reproduce here for your convenience. Please be sure to cite them as well, should you use their data.

## Models

Available [here](https://huggingface.co/misfitting/misfitting), but awaiting updates -- a few models missing at the moment.

## Citation

```
@inproceedings{
li2025misfitting,
title={(Mis)Fitting Scaling Laws: A Survey of Scaling Law Fitting Techniques in Deep Learning},
author={Margaret Li and Sneha Kudugunta and Luke Zettlemoyer},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=xI71dsS3o4}
}
```

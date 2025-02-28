# (Mis)Fitting Scaling Laws: A Survey of Scaling Laws

UNDER CONSTRUCTION
documentation coming

This repository contains the data and code to reproduce the figures and tables in the paper ["(Mis)Fitting Scaling Laws: A Survey of Scaling Laws"](https://arxiv.org/abs/2502.18969), by Margaret Li, Sneha Kudugunta and Luke Zettlemoyer.

## Training models

We've compiled all of the commands needed to download and preprocess data, and to train models, in one script:

```
bash scaling_laws/open_lm/ds/run_train_scale.sh
```
There are strings in this repo which need to be replaced with constants specific to your environment -- e.g., wandb username, slurm account and partition, or conda environment, if relevant. These can be found by grepping for `## TODO: FILL IN`.

## Data

Temporarily [here](https://drive.google.com/drive/folders/1hQCzv8sdppkihFZPTY1cLpiymJU0wqlP?usp=sharing), will be uploaded to github soon.

## Models

Available [here](https://huggingface.co/misfitting/misfitting), but awaiting updates -- a few models missing at the moment.

## Citation

BIBTEX TODO

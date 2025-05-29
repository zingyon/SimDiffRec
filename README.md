# SimDiffRec



fork by https://github.com/Tokkiu/ECL

## Introduction

SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation
The code is implemented based on DDPM, ECL-SR and CaDiRec.
Thank you for their valuable studies:)

Environment Dependencies
- Python
- Pytorch

## Dataset

Beauty, Toys and Sports(http://jmcauley.ucsd.edu/data/amazon/)

Yelp(https://www.yelp.com/dataset)

MovieLens(https://grouplens.org/datasets/movielen

## Run

```bash
python run_recbole.py --model=SimDiff --n_layers=2 --n_heads=2 --hidden_size=64 --inner_size=256 --hidden_act=gelu --initializer_range=0.02 --layer_norm_eps=1e-12 --attn_dropout_prob=0.2 --hidden_dropout_prob=0.2 --loss_type=CE --neg_sampling=None --mask_strategy=sample --mask_ratio=0.2 --encoder_loss_weight=1 --contrastive_loss_weight=0.001 --generate_loss_weight=0.2 --n_embedding=2 --n_sampling=5 --dataset=ml-1m --config_files=conf/config_d_ml-1m.yaml --gpu_id=1
```

### Hyperparameter

Hyperparameters for the proposed method are detailed in the paper, dataset-specific configurations are available in the `conf` directory, and all other parameters adhere to the default settings of the Recbole framework.

# Boosting Multi-Label Few-Shot Image Classification via Pairwise Feature Augmentation and Flexible Prompt Learning
Thanks for your attention. The following instructions can help you reproduce the experiments.

## Platform

Our experiments are conducted on a platform with NVIDIA GeForce RTX 3090.


## Set-up Experiment Environment

Our implementation is in Pytorch with python 3.7. The specific environment configuration can be found in ```environment.yml```.

## Datasets

- **MS COCO**: We include images from the official `train2014` and `val2014` splits.
- **PASCAL VOC**: We include images from the official `trainval` and `test` splits of VOC2007 and `trainval` of VOC2012. 



## Running

```
bash run_coco_all.sh
```
or
```
bash run_voc_all.sh
```

The detailed configurations can be found in the ```run_coco_all.sh```, ```run_voc_all.sh``` and ```opts.py```.


Some Args:  
- `dataset_config_file`: currently the code supports `configs/datasets/coco.yaml` and `configs/datasets/voc.yaml`.
- `lr`: learning rate.
- `n_ctx`: length of each prompt.
- `pool_size`: number of learnable prompts.
- `pfa_lr`: learning rate of the pairwise feature augmentation model.

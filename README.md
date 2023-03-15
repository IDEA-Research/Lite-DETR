# Lite-DETR
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=a-simple-framework-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/panoptic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/panoptic-segmentation-on-ade20k-val?p=a-simple-framework-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/instance-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/instance-segmentation-on-ade20k-val?p=a-simple-framework-for-open-vocabulary)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-simple-framework-for-open-vocabulary/instance-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/instance-segmentation-on-cityscapes-val?p=a-simple-framework-for-open-vocabulary)

This is the official implementation of the paper "[Lite DETR : An Interleaved Multi-Scale Encoder for Efficient DETR](https://arxiv.org/pdf/2303.07335.pdf)". Accepted to CVPR 2023.

Code will be released soon.
# Key Features
Efficient encoder design to reduce computational cost
- **Simple**. Dozens of lines code change (if not consider pluggable key-aware attention). 
- **Effective**. Reduce encoder cost by 50\% while preserve most of the orignal performance.
- **General**. Validated on a series of DETR models (Deformable DETR, H-DETR, DINO).

![hero_figure](figs/flops.png)
# Model Framework
![hero_figure](figs/framework.jpg)
# Results
Results on Deformable DETR
![hero_figure](figs/deformable.jpg)
Results on DINO and H-DETR
![hero_figure](figs/results.jpg)

[comment]: <> (![hero_figure]&#40;figs/vis.jpg&#41;)

[comment]: <> (# Bibtex)

[comment]: <> (If you find our work helpful for your research, please consider citing the following BibTeX entry.)

[comment]: <> (```bibtex)

[comment]: <> (```)

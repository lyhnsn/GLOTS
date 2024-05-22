# GLOTS
The codebase of "[Rethinking Transformers for Semantic Segmentation of Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/10209224)".

## Cite GLOTS

```
@ARTICLE{liu2023glots,
  author={Liu, Yuheng and Zhang, Yifan and Wang, Ye and Mei, Shaohui},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Rethinking Transformers for Semantic Segmentation of Remote Sensing Images}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2023.3302024}}
```

## Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation) for installation and [dataset_prepare.md](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Pre-trained Model

Download the [Model](https://drive.google.com/file/d/1yV070cXTrkCN2FTHKM2DIXI_dtVjaTJ6/view) and put it under the folder 'pretrain'.

## Data Preparation

Put the remote sensing datasets into `data/` directory as follows:
```
Vaihingen/
  ├── img_dir/
  │   ├── train/
  │   │     ├── 0001.jpg
  │   │     ├── 0002.jpg
  │   │     └── ...
  │   │
  │   └── val/
  │ 
  └── ann_dir/
      ├── train/
      │     ├── 0001.txt
      │     ├── 0002.txt
      │     └── ...
      │
      └── val/
```

## Usage

### train (example)

`python tools/train.py configs/beit/swin_beit-base_640x640_160k_vaihingen.py --gpu-id 0`

denotes "train GLOTS on vaihingen dataset on gpu 0".

### test (example)

`python tools/test.py work_dirs/swin_beit-base_640x640_160k_vaihingen/latest.pth --show-dir test_results/GLOTS --opacity 1 --gpu-id 0`

denotes "test the latest trained model of GLOTS on gpu 0, saving the prediction maps in 'test_results/GLOTS' folder, and the opacity of prediction maps is 100\% ".

## Reference

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main)

[BeiT](https://github.com/microsoft/unilm/tree/master/beit2)

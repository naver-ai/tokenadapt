# Forging Tokens for Improved Storage-efficient Training

Official Pytorch implementation of TokenAdapt | Paper

 [Minhyun Lee](https://scholar.google.com/citations?user=2hUlCnQAAAAJ&hl=ko) &nbsp; [Song Park](https://8uos.github.io/) &nbsp; [Byeongho Heo](https://sites.google.com/view/byeongho-heo/home) &nbsp; [Dongyoon Han](https://sites.google.com/site/dyhan0920/) &nbsp; [Hyunjung Shim](https://scholar.google.com/citations?user=KB5XZGIAAAAJ&hl=en) 

[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)

## Abstract

Recent advancements in Deep Neural Network (DNN) models have significantly improved performance across computer vision tasks. However, achieving highly generalizable and high-performing vision models requires extensive datasets, leading to large storage requirements. This storage challenge poses a critical bottleneck for scaling up vision models. Motivated by the success of discrete representations, SeiT proposes to use Vector-Quantized (VQ) feature vectors (i.e., tokens) as network inputs for vision classification. However, applying traditional data augmentations to tokens faces challenges due to input domain shift. To address this issue, we introduce TokenAdapt and ColorAdapt, simple yet effective token-based augmentation strategies. TokenAdapt realigns token embedding space for compatibility with spatial augmentations, preserving the model's efficiency without requiring fine-tuning. Additionally, ColorAdapt addresses color-based augmentations for tokens inspired by Adaptive Instance Normalization (AdaIN). We evaluate our approach across various scenarios, including storage-efficient ImageNet-1k classification, fine-grained classification, robustness benchmarks, and ADE-20k semantic segmentation. Experimental results demonstrate consistent performance improvement in diverse experiments, validating the effectiveness of our method.


## Usage

### Requirements
- Python3
- Pytorch (> 1.7)
- timm (0.5.4)

### Training
#### Downloading datasets
- The ImageNet-1k token datasets can be downloaded from [SeiT](https://github.com/naver-ai/seit/releases)
  - The tar file contains all the files required to train the ViT model; tokens, codebook and pre-defined token-synonym dictionary.
  - Download the file and place the extracted files under same directory for convinience
  
#### Training examples
- We used 8 V100 GPUs to train ViT-B with ImageNet-1k Tokens.
```
  DATA_DIR="path/data_dir"

  OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
      --model deit_base_token_32 \
      --codebook-path ${DATA_DIR}/codebook.ckpt \
      --train-token-file ${DATA_DIR}/imagenet1k-train-token.data.bin.gz \
      --train-label-file ${DATA_DIR}/imagenet1k-train-label.txt \
      --val-token-file ${DATA_DIR}/imagenet1k-val-token.data.bin.gz \
      --val-label-file ${DATA_DIR}/imagenet1k-val-label.txt \
      --token-synonym-dict ${DATA_DIR}/codebook-synonyms.json \
      --output_dir path/to/save \
      --batch-size 128 \
      --dist-eval \
      --tokenadapt-path path/to/tokenadapt.ckpt    
```  

### Evaluation

#### Preparing pre-trained weights
- The pre-trained weights of ViT models can be downloaded from [here](https://github.com/gaviotas/tokenadapt/releases/tag/v0.0).

#### Evaluation examples
```
DATA_DIR="path/data_dir"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model deit_base_token_32 \
    --codebook-path ${DATA_DIR}/codebook.ckpt \
    --train-token-file ${DATA_DIR}/imagenet1k-train-token.data.bin.gz \
    --train-label-file ${DATA_DIR}/imagenet1k-train-label.txt \
    --val-token-file ${DATA_DIR}/imagenet1k-val-token.data.bin.gz \
    --val-label-file ${DATA_DIR}/imagenet1k-val-label.txt \
    --token-synonym-dict ${DATA_DIR}/codebook-synonyms.json \
    --output_dir /mnt/tmp/log/test \
    --batch-size 128 \
    --dist-eval \
    --resume path/to/trained_vit.pth \
    --eval \
```

## Acknowledgements

This repository is heavily borrowed brom SeiT: [naver-ai/seit](https://github.com/naver-ai/seit).

## License

```
Copyright (c) 2023-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

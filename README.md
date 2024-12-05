# WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole Slide Images [MICCAI2024 oral]

=====
<details>
<summary>
    <b>WsiCaption: Multiple Instance Generation of Pathology Reports for Gigapixel Whole Slide Images</b>.
      <a href="https://arxiv.org/abs/2311.16480" target="blank">[Link]</a>
      <br><em>Pingyi Chen, Honglin Li, Chenglu Zhu, Sunyi Zheng, Lin Yang </em></br>
</summary>
</details>
 <b>Summary:</b>1. We propose a pipeline to curate high-quality WSI-text pairs from TCGA. The dataset <a href="https://drive.google.com/file/d/1KMvN8l7C8gUuD9Udl_NGlzEYR_A_nlQN/view?usp=drive_link" target="blank"><b>PathText</b></a> (for convenience, we now provide a  <a href="https://drive.google.com/file/d/1MLXUaqH5Yuv7RfyKW1hIWqnecHNqgZQR/view?usp=sharing" target="blank"><b>.json Version</b></a>  )contains about ten thousand pairs which will be publicly accessible. It can potentially promote the development of visual-language models in pathology. 2. We design a multiple instance generation framework(MI-Gen) (we provide a ResNet based <a href="https://drive.google.com/file/d/1BDT345Jh9iQWjaLyeWm_ioiOibxFwcR3/view?usp=sharing" target="blank"><b>ckpt</b></a> ). By incorporating the position-aware module, our model is more sensitive to the spatial information in WSIs.

<img src="pics/framework.png" width="1500px" align="center" />

## Pre-requisites:
We will share our collected slide-level captions but WSIs still need to be downloaded due to their large resolution.
[2024/12] We now provide the WSI features for convenience! The ResNet50 features on TCGA-BRCA is <a href="https://pan.baidu.com/s/1pAHuxJFAo80eA4Rd8RRuuQ?pwd=1mzw " target="blank"><b>here</b></a>: 
### Downloading TCGA Slides
To download diagnostic WSIs (formatted as .svs files), please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/).

### Processing Whole Slide Images
To process WSIs, first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained truncated ResNet50 is used to encode raw image patches into 1024-dim feature vectors, which we then save as .pt files for each WSI. We achieve the pre-processing of WSIs by using <a href="https://github.com/mahmoodlab/CLAM" target="blank"><b>CLAM</b></a>

### PathText: Slide-Text captions
We notice that TCGA includes scanning copies of pathology reports in the format of PDF1. But they are too long with redundant information and present in a complex structure. Therefore, we propose a pipeline to extract and clean pathological texts from TCGA, which can convert complex PDF files to concise WSI-text pairs with the assistance of large language models (LLM). We also use a classifier to remove the pairs with bad quality.

<img src="pics/DPT.png" width = "60%" height = "60%" alt="dataset construction" align=center />
 
Our dataset can be downloaded online now. The following folder structure is assumed for the PathText:
```bash
PathText/
    └──TCGA_BLCA/
        ├── case_1
              ├──annotation ##(slide-level captions we obtained by ocr and GPT)
              ├──case_1.pdf ##(softlink to the corresponding raw TCGA report)
              └── ...
        ├── case_2
        └── ...
    └──TCGA_BRCA/
        ├── case_1
        ├── case_2
        └── ...
    ...

TCGA-Slide-Features/
    └──TCGA_BLCA/
        ├── case_1.pt
        ├── case_2.pt
        └── ...
    └──TCGA_BRCA/
        ├── case_1.pt
        ├── case_2.pt
        └── ...
    ...
```
PathText contains the captions and TCGA-Slide-Features includes the extracted features of WSIs.

More details about the dataset are shown below. . (a) Histogram of text lengths. It shows that PathText includes
longer pathology reports compared to ARCH which only describes small patches.
(b) Word cloud showing 100 most frequent tokens.

<img src="pics/dataset.png" width = "60%" height = "60%" align="center" />

## Running Experiments
Experiments can be run using the following generic command-line:
### Training model
```shell
python main.py --mode 'Train' --n_gpu <GPUs to be used, e.g '0,1,2,3' for 4 cards training> --image_dir <SLIDE FEATURE PATH> --ann_path <CAPTION PATH> --split_path <PATH to the directory containing the train/val/test splits> 
```
### Testing model
```shell
python main.py --mode 'Test' --image_dir <SLIDE FEATURE PATH> --ann_path <CAPTION PATH> --split_path <PATH to the directory containing the train/val/test splits> --checkpoint_dir <PATH TO CKPT>
```

## Basic Environment
* Linux (Tested on Ubuntu 18.04) 
* NVIDIA GPU (Tested on Nvidia GeForce A100) with CUDA 12.0
* Python (3.8)
* PyTorch (1.10.0+cu111)
* torchvision (0.11.0+cu111)


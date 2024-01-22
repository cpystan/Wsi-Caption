# MI-Gen: Multiple Instance Generation of Pathology Reports for Gigapixel Whole Slide Images

=====
<details>
<summary>
    <b>MI-Gen: Multiple Instance Generation of Pathology Reports for Gigapixel Whole Slide Images</b>.
      <a href="https://arxiv.org/abs/2311.16480" target="blank">[Link]</a>
      <br><em>Pingyi Chen, Honglin Li, Chenglu Zhu, Sunyi Zheng, Lin Yang </em></br>
</summary>
</details>
 <b>Summary:</b>1. We propose a pipeline to curate high-quality WSI-text pairs from TCGA. The dataset (TCGA-PathoText) contains about ten thousand pairs which will be publicly accessible. It can potentially promote the development of visual-language models in pathology. 2. We design a multiple instance generation framework(MI-Gen). By incorporating the position-aware module, our model is more sensitive to the spatial information in WSIs.

<img src="pics/framework.png" width="1500px" align="center" />

## Pre-requisites:
We will share our collected slide-level captions but WSIs still need to be downloaded due to their large resolution.
### Downloading TCGA Slides
To download diagnostic WSIs (formatted as .svs files), please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/).

### Processing Whole Slide Images
To process WSIs, first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained truncated ResNet50 is used to encode raw image patches into 1024-dim feature vectors, which we then save as .pt files for each WSI. We achieve the pre-processing of WSIs by using <a href="https://github.com/mahmoodlab/CLAM" target="blank"><b>CLAM</b></a>

### TCGA-PathoText: Slide-Text captions

<img src="pics/dpt.png" height="500px" width="1000px" align="center" />

Our dataset can be downloaded online now. The following folder structure is assumed for the TCGA-PathoText:
```bash
DATA_ROOT_DIR/
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
    └──TCGA_GBMLGG/
        ├── case_1
        ├── case_2
        └── ...
    └──TCGA_LUAD/
        ├── case_1
        ├── case_2
        └── ...
    └──TCGA_UCEC/
        ├── case_1
        ├── case_2
        └── ...
    ...
```
DATA_ROOT_DIR is the base directory of all datasets / cancer type(e.g. the directory to your SSD).

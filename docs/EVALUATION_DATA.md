# Preparing evaluation datasets

This readme provides instructions on downloading the downstream datasets used for zero-shot evaluation. 

**About the evaluation datasets:** For evaluating the zero-shot performance of UniMed-CLIP and prior Medical VLMs, we utilize 21 medical datasets that covers 6 diverse modalities. We refer the readers to Table 5 (supplementary material), for additional details about evaluation datasets.

To facilitate quick prototyping, we directly provide the processed evaluation datasets. Download the datasets (zipped form) [from this link](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EdaUYopuq6lLhOTl0by1A9oBs92y56tV1g4iins9QiFwVg?e=WhKzUd). 


After downloading and extracting the files, the directory structure looks like below.
```
extracted_folder/
|–– LC25000_test/
|–– BACH/
|–– chexpert/
|–– organs_coronal/
|–– thyroid_us/
|–– skin_tumor/
|–– organs_sagittal/
|–– fives_retina/
|–– meniscal_mri/
# remaining datasets...
```


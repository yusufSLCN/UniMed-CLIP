# UniMed-CLIP: Towards a Unified Image-Text Pretraining Paradigm for Diverse Medical Imaging Modalities

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>

> [Muhammad Uzair Khattak*](https://muzairkhattak.github.io/), [
Shahina Kunhimon*](https://scholar.google.com/citations?hl=en&user=yYPksIkAAAAJ), [Muzammal Naseer](https://muzammal-naseer.com/), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://sites.google.com/view/fahadkhans/home)

**Mohamed bin Zayed University of AI, Swiss Federal Institute of Technology Lausanne (EPFL), Khalifa University,  Australian National University, Linköping University**

*Equally contributing first authors

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.10372)
[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)](docs/UniMed-DATA.md)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/UzairK/unimed-clip-medical-image-zero-shot-classification)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](getting_started_unimed_clip.ipynb)

This repository contains the code implementation for UniMed-CLIP, a family of strong Medical Contrastive VLMs trained on the proposed UniMed-dataset. We further provide detailed instructions and annotation files for preparing UniMed Dataset for promoting open-source practices in advancing Medical VLMs.

---


# Updates
* **Dec 22, 2024**
  * Demo released on Hugging Face spaces ([view demo](https://huggingface.co/spaces/UzairK/unimed-clip-medical-image-zero-shot-classification))
* **Dec 13, 2024**
  * Annotations and code scripts for preparing the UniMed pretraining dataset are released.
  * UniMed-CLIP training and inference code are released, along with pretrained checkpoints.
---

# Highlights

![main figure](docs/teaser_photo.svg)


> **<p align="justify"> Abstract:** *Vision-Language Models (VLMs) trained via contrastive learning have achieved
> notable success in natural image tasks. However, their application in the medical domain remains limited due to
> the scarcity of openly accessible, large-scale medical image-text datasets. Existing medical VLMs either train on
> closed-source proprietary or relatively small open-source datasets that do not generalize well. Similarly, most models
> remain specific to a single or limited number of medical imaging domains, again restricting their applicability to other
> modalities. To address this gap, we introduce UniMed, a large-scale, open-source multi-modal medical dataset comprising
> over 5.3 million image-text pairs across six diverse imaging modalities: X-ray, CT, MRI, Ultrasound, Pathology, and Fundus.
> UniMed is developed using a data-collection framework that leverages Large Language Models (LLMs) to transform 
> modality-specific classification datasets into image-text formats while incorporating existing image-text data 
> from the medical domain, facilitating scalable VLM pretraining. Using UniMed, we trained UniMed-CLIP, a unified VLM 
> for six modalities that significantly outperforms existing generalist VLMs and matches modality-specific medical VLMs,
> achieving notable gains in zero-shot evaluations. For instance, UniMed-CLIP improves over BiomedCLIP (trained on proprietary data)
> by an absolute gain of +12.61, averaged over 21 datasets, while using 3x less training data. 
> To facilitate future research, we release UniMed dataset, training codes, and models.* </p>

### UniMed-CLIP: Open-source Contrastive Medical VLMs excelling across 6 diverse medical modalities

Main contributions of our work are:
1) **UniMed Dataset: An open-source, large-scale medical multi-modal dataset:** We develop UniMed using an LLM-in-the-loop framework, comprising over 5.3 million samples. It covers six diverse medical modalities and provides a robust foundation for training generalizable medical VLMs.
2) **UniMed-CLIP VLMs:**  Building upon UniMed, we train a family of contrastive VLMs which that significantly outperforms existing generalist VLMs and often matches modality-specific specialist VLMs.
3) **Extensive evaluations and analysis:** We ablate on different design choices while developing both the UniMed pretraining dataset and UnMed-CLIP VLMs. Furthermore, our training code, dataset, and model checkpoints are open-sourced to encourage further progress in medical VLMs.


| Method      | Paper    | X-ray | Retinal-Fundus | CT     | MRI  | US   | Histopathology | Avg.      |
|-------------|----------|-------|----------------|--------|------|------|-----------|-----------|
| BioMedCLIP  | [Link](https://arxiv.org/abs/2303.00915) |   55.43 |   22.87        | 43.99  |    64.59  |   49.20   |  54.50 | 49.02     |
| PMC-CLIP    | [Link](https://arxiv.org/abs/2303.07240) |   52.64    |    25.84            | 66.06  | 63.68     |  62.51    |      53.56     | 53.37     |
| UniMed-CLIP | [Link](https://arxiv.org/abs/2412.10372) |     **68.78**  |    **31.23**      |  **85.54** |  **68.83**    |    **68.64**  |  **59.96**    | **61.63** |


## Quick Links
  - [Installation](#installation)
  - [Quick Start](#quick-start-for-inference-with-unimed-clip-models)
  - [Pre-trained Models](#pre-trained-models)
  - [Preparing UniMed-Dataset](#preparing-unimed-dataset)
  - [Training UniMed-CLIP](#bugs-or-questions)
  - [Evaluating UniMed-CLIP](#evaluating-unimed-clip)
  - [Questions and Support](#questions-and-support)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgement)

## Installation

Before using UniMed-CLIP for training and inference, please refer to the installation instructions described at [INSTALL.md](docs/INSTALL.md)

## Quick Start for inference with UniMed-CLIP models 

We provide an online [hugging-face demo](https://huggingface.co/spaces/UzairK/unimed-clip-medical-image-zero-shot-classification) and [jupyter notebook](getting_started_unimed_clip.ipynb) for using pretrained UniMed-CLIP models for zero-shot classification across 6 diverse medical modalities. Additionally, we provide a sample code below to get started.


```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
os.chdir(src_path)

from open_clip import create_model_and_transforms, get_mean_std, HFTokenizer
from PIL import Image
import torch

# Define main parameters
model_name = 'ViT-B-16-quickgelu' # available pretrained weights ['ViT-L-14-336-quickgelu', 'ViT-B-16-quickgelu']
pretrained_weights = "./unimed_clip_vit_b16.pt" # Path to pretrained weights
text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" # available pretrained weights ["microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"]
mean, std = get_mean_std()
device='cuda'
# Load pretrained model with transforms
model, _, preprocess = create_model_and_transforms(
    model_name,
    pretrained_weights,
    precision='amp',
    device=device,
    force_quick_gelu=True,
    mean=mean, std=std,
    inmem=True,
    text_encoder_name=text_encoder_name,)

tokenizer = HFTokenizer(
    text_encoder_name,
    context_length=256,
    **{},)

# Prepare text prompts using different class names
text_prompts = ['CT scan image displaying the anatomical structure of the right kidney.',
                'pneumonia is indicated in this chest X-ray image.', 
                'this is a MRI photo of a brain.', 
                'this fundus image shows optic nerve damage due to glaucoma.',
                'a histopathology slide showing Tumor',
                "Cardiomegaly is evident in the X-ray image of the chest."]
texts = [tokenizer(cls_text).to(next(model.parameters()).device, non_blocking=True) for cls_text in text_prompts]
texts = torch.cat(texts, dim=0)

# Load and preprocess images
test_imgs = [
    'brain_MRI.jpg',
    'ct_scan_right_kidney.tiff',
    'tumor_histo_pathology.jpg',
    'retina_glaucoma.jpg',
    'xray_cardiomegaly.jpg',
    'xray_pneumonia.png',
]
images = torch.stack([preprocess(Image.open(("../docs/sample_images" + img))) for img in test_imgs]).to(device)

# Inference
with torch.no_grad():
    text_features = model.encode_text(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = model.encode_image(images)
    logits = (image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()

# Print class probabilities for each image
top_k = -1

for i, img in enumerate(test_imgs):
    pred = text_prompts[sorted_indices[i][0]]

    top_k = len(text_prompts) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{text_prompts[jth_index]}: {logits[i][jth_index]}')
    print('\n')
```

<details>
<summary>Outputs</summary>

```python
brain_MRI.jpg:
----
this is a MRI photo of a brain: 0.9981486797332764
Cardiomegaly is evident in the X-ray image of the chest.: 0.0011040412355214357
CT scan image displaying the anatomical structure of the right kidney.: 0.00034158056951127946
pneumonia is indicated in this chest X-ray image.: 0.00014067212759982795
this fundus image shows optic nerve damage due to glaucoma.: 0.0001399167231284082
a histopathology slide showing Tumor: 0.00012514453555922955


ct_scan_right_kidney.tiff:
----
CT scan image displaying the anatomical structure of the right kidney.: 0.9825534224510193
a histopathology slide showing Tumor: 0.013478836975991726
this is a MRI photo of a brain: 0.003742802422493696
Cardiomegaly is evident in the X-ray image of the chest.: 0.00010370105155743659
this fundus image shows optic nerve damage due to glaucoma.: 6.942308391444385e-05
pneumonia is indicated in this chest X-ray image.: 5.183744360692799e-05


tumor_histo_pathology.jpg:
----
a histopathology slide showing Tumor: 0.9301006197929382
this is a MRI photo of a brain: 0.0670388713479042
pneumonia is indicated in this chest X-ray image.: 0.001231830450706184
this fundus image shows optic nerve damage due to glaucoma.: 0.0008338663610629737
Cardiomegaly is evident in the X-ray image of the chest.: 0.0006468823994509876
CT scan image displaying the anatomical structure of the right kidney.: 0.0001478752092225477


retina_glaucoma.jpg:
----
this fundus image shows optic nerve damage due to glaucoma.: 0.9986233711242676
Cardiomegaly is evident in the X-ray image of the chest.: 0.0009356095688417554
pneumonia is indicated in this chest X-ray image.: 0.0003371761704329401
this is a MRI photo of a brain: 8.056851947912946e-05
a histopathology slide showing Tumor: 1.5897187040536664e-05
CT scan image displaying the anatomical structure of the right kidney.: 7.302889116544975e-06


xray_cardiomegaly.jpg:
----
Cardiomegaly is evident in the X-ray image of the chest.: 0.9992433786392212
pneumonia is indicated in this chest X-ray image.: 0.00038846206734888256
this is a MRI photo of a brain: 0.00034906697692349553
a histopathology slide showing Tumor: 9.712741302791983e-06
this fundus image shows optic nerve damage due to glaucoma.: 5.269657776807435e-06
CT scan image displaying the anatomical structure of the right kidney.: 4.128277396375779e-06


xray_pneumonia.png:
----
pneumonia is indicated in this chest X-ray image.: 0.9995973706245422
Cardiomegaly is evident in the X-ray image of the chest.: 0.0003444128960836679
this fundus image shows optic nerve damage due to glaucoma.: 3.1277508242055774e-05
this is a MRI photo of a brain: 1.2267924830666743e-05
a histopathology slide showing Tumor: 1.0387550901214126e-05
CT scan image displaying the anatomical structure of the right kidney.: 4.427450676303124e-06
```
</details>

**Gradio Demo:**
Additionally, you can run local-gradio demo by running the `python app.py` command. This will automatically set-up the demo (including pretrained weights downloading).

## Pre-trained Models

We provide 3 model weights for UniMed-CLIP as listed in the table below. For larger vision models (ViT Large variant), we found utilizing base size text encoder as the optimal choice for effective downstream zero-shot performance.

| `model_name`       | `text encoder`                              |                                                                               `pretrained_weights`                                                                               | Res. |      GPUs       | Avg. score on 21 datasets |
|:-------------------|:--------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----:|:---------------:|:-------------------------:|
| ViT-B-16-quickgelu | BiomedNLP-BiomedBERT-base-uncased-abstract  |          [`unimed_clip_vit_b16`](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ee8EpjZS6SJGiZUrV7DyLxkBrVFir5YzMjYZIc8aEc2oUA?e=I7KvRb)           | 224  | 16 x A100 (40G) |           61.63           |
| ViT-L-14-quickgelu | BiomedNLP-BiomedBERT-large-uncased-abstract | [`unimed_clip_vit_l14_large_text_encoder`](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/Ea2CZ1dc_B9PsTHp5kAeUBsB-bncfRmjra63YDM0bn9JRw?e=hWxW1s) | 336  | 16 x A100 (40G) |           62.09           |
| ViT-L-14-quickgelu | BiomedNLP-BiomedBERT-base-uncased-abstract  | [`unimed_clip_vit_l14_base_text_encoder`](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/uzair_khattak_mbzuai_ac_ae/EfeUk8TpOkNEsRDMIIYlxNcB_8swAJgt0Ix3igjxM2z_nw?e=3CnPKb)  | 336  | 16 x A100 (40G) |           64.84           |


## Preparing UniMed-Dataset

For preparing UniMed pretraining dataset, we provide instructions for i) Downloading raw datasets from publicly available sources and ii) downloading processed annotations and merging with raw-datasets to build UniMed-CLIP 
Refer to the detailed instructions described in [UniMed-DATA.md](docs/UniMed-DATA.md).


## Training UniMed-CLIP

For training UniMed-CLIP, we provide different model configs in the `run_configs_400m.py`. Make sure to set the required parameters in the config file (e.g., dataset paths). 

We initialize image encoder and text encoder weights using MetaCLIP and BiomedBERT (uncased-abstract) models respectively. For example, to train UniMed-CLIP ViT-B/16 on UniMed dataset, run the following command:

- Training UniMed-CLIP on a single node with 8 GPUs
```
# first download weights for metaclip, weights for BiomedBERT will be downloaded and loaded automatically
wget https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt
# running on a single node with 8 80GB-A100 GPUs
torchrun --nproc_per_node=8 src/training/main.py b16_400m <experiment-name> <path/to/b16_400m.pt>
```

- Training UniMed-CLIP using multiple nodes 4 GPUs per node
```
# first download weights for metaclip, weights for BiomedBERT will be downloaded and loaded automatically
# first download weights for metaclip, weights for BiomedBERT will be downloaded and loaded automatically
# running on multi-nodes (4 nodes with 4 40GB-A100 GPUs)
python submitit_openclip.py b16_400m <path/to/b16_400m.pt> <experiment-name> --partition <gpu_partition> --nodes 4 --ngpus 4 --max_job_time "1-24:00:00"
```

## Evaluating UniMed-CLIP

We provide instructions for performing zero-shot evaluation using pretrained UniMed-CLIP models. 

- First download test datasets using instructions provided in [EVALUATION_DATA.md](docs/EVALUATION_DATA.md), and set-up dataset paths in `clipeval/dataset_catalog.json`.
- Run the following command to evaluate UniMed-CLIP zero-shot performance on 21 medical datasets
```
torchrun --nproc_per_node=1 src/training/main.py b16_400m_eval <logs-path-to-save> <path-to-pretrained-weights>
```

## Questions and Support

Contact Muhammad Uzair (uzair.khattak@mbzuai.ac.ae) or Shahina Kunhimon (shahina.kunhimon@mbzuai.ac.ae) regarding any questions about the code and the paper.


## Citation

If you find our work and this repository helpful, please consider giving our repo a star and citing our paper as follows:

```bibtex
@article{khattak2024unimed,
  title={UniMed-CLIP: Towards a Unified Image-Text Pretraining Paradigm for Diverse Medical Imaging Modalities},
  author={Khattak, Muhammad Uzair and Kunhimon, Shahina and Naseer, Muzammal and Khan, Salman and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:2412.10372},
  year={2024}
}
```

## Acknowledgement
Our code repository is mainly built on [MetaCLIP](https://github.com/facebookresearch/MetaCLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip). We thank the authors for releasing their code. 

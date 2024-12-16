# Copyright (c) Meta Platforms, Inc. and affiliates

# usage:
# torchrun --nproc_per_node=4 src/training/main.py b16_400m my-experiment-name <path-to-metaclip-pretrained-checkpoint>
from dataclasses import dataclass
from configs import Config


@dataclass
class b32_400m(Config):
    inmem=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    # First prepare UniMed-Dataset using instructions in the docs/PREPARE-UniMed-DATA.md and then,
    # provide paths for each sub-part of UniMed dataset below.
    train_data="/<dataset-path>/radimagenet_webdataset/dataset-{000001..001049}.tar::/<dataset-path>/chexpert_webdataset/dataset-{000001..000212}.tar::/<dataset-path>/openi_webdataset/dataset-{000001..000007}.tar::/<dataset-path>/chest_xray8_webdataset/dataset-{000001..000113}.tar::/<dataset-path>/mimic_cxr/dataset-{000001..000270}.tar::/<dataset-path>/roco_webdataset/dataset-{000001..000061}.tar::/<dataset-path>/pmc_clip_webdataset/dataset-{000001..001645}.tar::/<dataset-path>/llava_med_alignment_set_webdataset/dataset-{000001..000468}.tar::/<dataset-path>/llava_med_hq_60k_set_webdataset/dataset-{000001..000265}.tar::/<dataset-path>/quilt_webdataset/dataset-{000001..001018}.tar::/<dataset-path>/retina_part1_webdataset/dataset-{000001..000155}.tar::/<dataset-path>/retina_part2_webdataset/dataset-{000001..000013}.tar::/<dataset-path>/retina_part3_webdataset/dataset-{000001..000006}.tar"
    # train_num_samples = 1049000 (radimagenet) + 212000 (chexpert) + 7000 (openi) + 113000 (chest-xray8) + 270000 (mimic-cxr) + 61000 (rocov2) + 1645000 (pmc-clip) + 468000 (llavamed-alignment) + 265000 (llava-medhq) + 1018000 (quilt) + 155000 (retina part 1) + 13000 (retina part 2) + 6000 (retina part 3)
    # Total training samples must equal total dataset size
    train_num_samples = 5282000
    # By default, we provide equal weightage to all dataset parts
    train_data_upsampling_factors = "1::1::1::1::1::1::1::1::1::1::1::1"
    # ----------------------------------------
    workers=8
    batch_size=128
    epochs= 10
    eval_freq = 1
    model="ViT-B-32-quickgelu"
    name="ViT-B-32"
    force_quick_gelu=True
    warmup=2000
    seed=0
    local_loss=True
    gather_with_grad=True
    nodes=16
    ngpus=4
    imagenet_val = None
    report_to = 'wandb'
    tokenizer_context_length = 256
    text_encode_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

@dataclass
class b32_400m_eval(Config):
    inmem=True
    engine="train_one_epoch_ex"
    eval_steps=5000
    save_frequency=1
    train_data=""
    workers=8
    eval_freq = 1
    train_num_samples=400000000
    batch_size=512
    epochs=10
    model="ViT-B-32-quickgelu"
    name="ViT-B-32"
    force_quick_gelu=True
    warmup=2000
    seed=0
    local_loss=True
    gather_with_grad=True
    nodes=16
    ngpus=4
    imagenet_val = None
    pretrained = '<path-to-metaclip-pretrained-weights-file>/b16_400m.pt'
    text_encode_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    tokenizer_context_length = 256


@dataclass
class b16_400m(b32_400m):
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    grad_checkpointing=True
    # Change below
    pretrained = '<path-to-metaclip-pretrained-weights-file>/b16_400m.pt'
    text_encode_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

@dataclass
class b16_400m_eval(b32_400m_eval):
    model="ViT-B-16-quickgelu"
    name="ViT-B-16"
    grad_checkpointing=True
    pretrained = '<path-to-metaclip-pretrained-weights-file>/b16_400m.pt'
    text_encoder_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'


@dataclass
class l14_400m(b32_400m): 
    model="ViT-L-14-336-quickgelu"
    name="ViT-L-14"
    lr=0.0004
    grad_checkpointing=True
    batch_size=128
    nodes=16
    ngpus=8
    text_encoder_model_name = 'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract'

@dataclass
class l14_400m_eval(b32_400m_eval):
    model="ViT-L-14-336-quickgelu"
    name="ViT-L-14"
    lr=0.0004
    grad_checkpointing=True
    batch_size=256
    nodes=16
    ngpus=8
    text_encoder_model_name = 'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract'

@dataclass
class l14_400m_base_text_encoder(b32_400m):
    model="ViT-L-14-336-quickgelu"
    name="ViT-L-14"
    lr=0.0004
    grad_checkpointing=True
    batch_size=128
    nodes=16
    ngpus=8
    text_encoder_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

@dataclass
class l14_400m_base_text_encoder_eval(b32_400m_eval):
    model="ViT-L-14-336-quickgelu"
    name="ViT-L-14"
    lr=0.0004
    grad_checkpointing=True
    batch_size=256
    nodes=16
    ngpus=8
    text_encode_model_name = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

if __name__ == "__main__":
    import inspect
    import sys
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj):
            print(name)

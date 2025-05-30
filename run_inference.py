import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
os.chdir(src_path)

from src.open_clip import create_model_and_transforms, get_mean_std, HFTokenizer
from PIL import Image
import torch

# Define main parameters
model_name = 'ViT-B-16-quickgelu' # available pretrained weights ['ViT-L-14-336-quickgelu', 'ViT-B-16-quickgelu']
pretrained_weights = "/scratch/local/radiologie/salcan/llava-priv/unimed-clip/unimed_clip_vit_b16.pt" # Path to pretrained weights
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
sample_images_dir = "/scratch/local/radiologie/salcan/llava-priv/unimed-clip/docs/sample_images/"
images = torch.stack([preprocess(Image.open((sample_images_dir + img))) for img in test_imgs]).to(device)

# Inference
with torch.no_grad():
    text_features = model.encode_text(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = model.encode_image(images)
    print('Image features shape:', image_features.shape)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = (model.logit_scale.exp() * image_features @ text_features.t()).detach().softmax(dim=-1)
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
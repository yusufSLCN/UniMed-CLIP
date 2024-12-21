
import gradio as gr
# Switch path to root of project
import os
import sys
# Get the current working directory
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
os.chdir(src_path)
# Add src directory to sys.path
sys.path.append(src_path)
from open_clip import create_model_and_transforms
from huggingface_hub import hf_hub_download
from open_clip import HFTokenizer
import torch

class create_unimed_clip_model:
    def __init__(self, model_name):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
        std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
        if model_name == "ViT/B-16":
            # Download the weights
            weights_path = hf_hub_download(
                repo_id="UzairK/unimed-clip-vit-b16",
                filename="unimed-clip-vit-b16.pt"
            )
            self.pretrained = weights_path  # Path to pretrained weights
            self.text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
            self.model_name = "ViT-B-16-quickgelu"
        elif model_name == 'ViT/L-14@336px-base-text':
            # Download the weights
            self.model_name = "ViT-L-14-336-quickgelu"
            weights_path = hf_hub_download(
                repo_id="UzairK/unimed_clip_vit_l14_base_text_encoder",
                filename="unimed_clip_vit_l14_base_text_encoder.pt"
            )
            self.pretrained = weights_path  # Path to pretrained weights
            self.text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        self.tokenizer = HFTokenizer(
            self.text_encoder_name,
            context_length=256,
            **{},
        )
        self.model, _, self.processor = create_model_and_transforms(
            self.model_name,
            self.pretrained,
            precision='amp',
            device=self.device,
            force_quick_gelu=True,
            pretrained_image=False,
            mean=mean, std=std,
            inmem=True,
            text_encoder_name=self.text_encoder_name,
        )

    def __call__(self, input_image, candidate_labels, hypothesis_template):
        # Preprocess input
        input_image = self.processor(input_image).unsqueeze(0).to(self.device)
        if hypothesis_template == "":
            texts = [
                self.tokenizer(cls_text).to(self.device)
                for cls_text in candidate_labels
            ]
        else:
            texts = [
                self.tokenizer(hypothesis_template + " " + cls_text).to(self.device)
                for cls_text in candidate_labels
            ]
        texts = torch.cat(texts, dim=0)
        # Perform inference
        with torch.no_grad():
            text_features = self.model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = self.model.encode_image(input_image)
            logits = (image_features @ text_features.t()).softmax(dim=-1).cpu().numpy()
            return {cls_text: float(score) for cls_text, score in zip(candidate_labels, logits[0])}

pipes = {
    "ViT/B-16": create_unimed_clip_model(model_name="ViT/B-16"),
    "ViT/L-14@336px-base-text": create_unimed_clip_model(model_name='ViT/L-14@336px-base-text'),
}
# Define Gradio inputs and outputs
inputs = [
    gr.Image(type="pil", label="Image", width=300, height=300),
    gr.Textbox(label="Candidate Labels (comma-separated)"),
    gr.Radio(
        choices=["ViT/B-16", "ViT/L-14@336px-base-text"],
        label="Model",
        value="ViT/B-16",
    ),
    gr.Textbox(label="Prompt Template", placeholder="Optional prompt template as prefix",
               value=""),
]
outputs = gr.Label(label="Predicted Scores")

def shot(image, labels_text, model_name, hypothesis_template):
    labels = [label.strip(" ") for label in labels_text.strip(" ").split(",")]
    res = pipes[model_name](input_image=image,
           candidate_labels=labels,
           hypothesis_template=hypothesis_template)
    return {single_key: res[single_key] for single_key in res.keys()}
# Define examples

examples = [
    ["../docs/sample_images/brain_MRI.jpg", "CT scan image displaying the anatomical structure of the right kidney., pneumonia is indicated in this chest X-ray image., this is a MRI photo of a brain., this fundus image shows optic nerve damage due to glaucoma., a histopathology slide showing Tumor, Cardiomegaly is evident in the X-ray image of the chest.", "ViT/B-16", ""],
    ["../docs/sample_images/ct_scan_right_kidney.jpg",
     "CT scan image displaying the anatomical structure of the right kidney., pneumonia is indicated in this chest X-ray image., this is a MRI photo of a brain., this fundus image shows optic nerve damage due to glaucoma., a histopathology slide showing Tumor, Cardiomegaly is evident in the X-ray image of the chest.",
     "ViT/B-16", ""],
    ["../docs/sample_images/retina_glaucoma.jpg",
     "CT scan image displaying the anatomical structure of the right kidney., pneumonia is indicated in this chest X-ray image., this is a MRI photo of a brain., this fundus image shows optic nerve damage due to glaucoma., a histopathology slide showing Tumor, Cardiomegaly is evident in the X-ray image of the chest.",
     "ViT/B-16", ""],
    ["../docs/sample_images/tumor_histo_pathology.jpg",
     "CT scan image displaying the anatomical structure of the right kidney., pneumonia is indicated in this chest X-ray image., this is a MRI photo of a brain., this fundus image shows optic nerve damage due to glaucoma., a histopathology slide showing Tumor, Cardiomegaly is evident in the X-ray image of the chest.",
     "ViT/B-16", ""],
    ["../docs/sample_images/xray_cardiomegaly.jpg",
     "CT scan image displaying the anatomical structure of the right kidney., pneumonia is indicated in this chest X-ray image., this is a MRI photo of a brain., this fundus image shows optic nerve damage due to glaucoma., a histopathology slide showing Tumor, Cardiomegaly is evident in the X-ray image of the chest.",
     "ViT/B-16", ""],
    ["../docs/sample_images//xray_pneumonia.png",
     "CT scan image displaying the anatomical structure of the right kidney., pneumonia is indicated in this chest X-ray image., this is a MRI photo of a brain., this fundus image shows optic nerve damage due to glaucoma., a histopathology slide showing Tumor, Cardiomegaly is evident in the X-ray image of the chest.",
     "ViT/B-16", ""],
]

iface = gr.Interface(shot,
            inputs,
            outputs,
            examples=examples,
            description="""<p>Demo for UniMed CLIP, a family of strong Medical Contrastive VLMs trained on UniMed-dataset. For more information about our project, refer to our paper and github repository. <br>
            Paper: <a href='https://arxiv.org/abs/2412.10372'>https://arxiv.org/abs/2412.10372</a> <br>
            Github: <a href='https://github.com/mbzuai-oryx/UniMed-CLIP'>https://github.com/mbzuai-oryx/UniMed-CLIP</a> <br><br>
            <b>[DEMO USAGE]</b> To begin with the demo, provide a picture (either upload manually, or select from the given examples) and class labels. Optionally you can also add template as an prefix to the class labels. <br> <b>[NOTE]</b> This demo is running on CPU and thus the response time might be a bit slower. Running it on a machine with a GPU will result in much faster predictions. </p>""",

            title="Zero-shot Medical Image Classification with UniMed-CLIP")

iface.launch(allowed_paths=["/home/user/app/docs/sample_images"])

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""refactored from `main` in `eval_zeroshot.py` (SLIP) for clarity.
"""
import random
import torch
import json
import os
from tqdm import tqdm
from sklearn import metrics
from constants import RSNA_CLASS_PROMPTS_webdataset, modality_indices_radimagenet_test_set
from collections import defaultdict


def load_metadata(metadir="clipeval"):
    with open(os.path.join(metadir, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    with open(os.path.join(metadir, 'templates.json')) as f:
        all_templates = json.load(f)

    with open(os.path.join(metadir, 'labels.json')) as f:
        all_labels = json.load(f)
    return catalog, all_templates, all_labels


def evaluate(d, val_loader, templates, labels, model, tokenizer, classnorm=False):
    print('Evaluating {}'.format(d))

    is_acc = d not in ['FGVCAircraft', 'OxfordPets', 'Caltech101', 'Flowers102', 'Kinetics700', 'HatefulMemes']
    if d == 'radimagenet':
        acc, us_acc, mri_acc, ct_acc = validate_zeroshot(val_loader, templates, labels, model, tokenizer,
                                                         is_acc, d, classnorm)
    else:
        acc_or_outputs = validate_zeroshot(val_loader, templates, labels, model, tokenizer, is_acc, d, classnorm)
    if d in ['FGVCAircraft', 'OxfordPets', 'Caltech101', 'Flowers102']:
        metric = mean_per_class(*acc_or_outputs)
    elif d == 'Kinetics700':
        top1, top5 = accuracy(*acc_or_outputs, topk=(1, 5))
        metric = (top1 + top5) / 2
        metric = metric.item()
    elif d == 'HatefulMemes':
        metric = roc_auc(*acc_or_outputs)
    elif d == 'radimagenet':
        metric = {"acc": acc, "US acc": us_acc, "MRI acc": mri_acc, "CT acc": ct_acc}
    else:
        metric = acc_or_outputs

    return metric


@torch.no_grad()
def build_text_features(templates, labels, model, tokenizer, skip_text_projection=False, classnorm=False):
    # TODO: add device
    text_features = []
    if type(templates) == dict:
        class_similarities = []
        class_names = []
        for cls_name, cls_text in templates.items():
            texts = tokenizer(cls_text).to(next(model.parameters()).device, non_blocking=True)
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            if True:
                cls_sim = class_embeddings.mean(dim=0)  # equivalent to prompt ensembling
            else:
                cls_sim = class_embeddings[0]
            class_similarities.append(cls_sim)
            class_names.append(cls_name)

        text_features = torch.stack(class_similarities, dim=0)
    elif type(templates) == list and templates[0] == "Meniscal abnormality detected in MRI imaging of the knee.":
        print("Encoding captions for RadImageNet dataset")
        for single_template in templates:
            texts = tokenizer(single_template).to(next(model.parameters()).device, non_blocking=True)
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0).squeeze(1)
    else:
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]

            texts = tokenizer(texts).to(next(model.parameters()).device, non_blocking=True)
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)
    mean, std = None, None
    if classnorm:
        mean, std = text_features.mean(dim=0)[None, :], text_features.std(dim=0)[None, :]
        text_features = (text_features - mean) / std
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features, mean, std


def generate_chexpert_class_prompts(class_prompts, n=None):
    """Generate text prompts for each CheXpert classification task
    Parameters
    ----------
    n:  int
        number of prompts per class
    Returns
    -------
    class prompts : dict
        dictionary of class to prompts
    """

    prompts = {}
    for k, v in class_prompts.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        # randomly sample n prompts for zero-shot classification
        # TODO: we shall make use all the candidate prompts for autoprompt tuning
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts


def generate_rsna_class_prompts(class_prompts, n=None):
    prompts = {}
    for k, v in class_prompts.items():
        cls_prompts = []
        keys = list(v.keys())

        for k0 in v[keys[0]]:
            for k1 in v[keys[1]]:
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        # randomly sample n prompts for zero-shot classification
        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
        print(f'sample {len(prompts[k])} num of prompts for {k} from total {len(cls_prompts)}')
    return prompts


@torch.no_grad()
def validate_zeroshot(val_loader, templates, labels, model, tokenizer, is_acc, name, classnorm=False):
    # switch to evaluate mode
    model.cuda()
    model.eval()

    total_top1 = 0
    total_images = 0

    all_outputs = []
    all_targets = []

    text_features = None
    # Initialize per-class accuracy variables
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for samples in tqdm(val_loader):
        # Below if will run only for one iteration
        if text_features is None:
            print('=> encoding captions')
            if name == "chexpert-5x200":
                if not type(templates[list(templates.keys())[0]]) == list:
                    prompted_templates = generate_chexpert_class_prompts(templates, 10)  # 10 prompts per class
                else:
                    k = 11  # This means all 10 templates to be used...
                    print(f"Using {k - 1} templates for the ensembling at test time")
                    for single_key in templates.keys():
                        templates[single_key] = templates[single_key][0:k]
                    prompted_templates = templates
                text_features, mean, std = build_text_features(prompted_templates, None, model, tokenizer,
                                                               classnorm=classnorm)
            elif name == "rsna_pneumonia":
                if not type(templates[list(templates.keys())[0]]) == list:
                    temp = generate_rsna_class_prompts(templates, 10)  # 10 prompts per class
                    # For the case of Pneumonia, we also need second template for normal as well
                    prompted_templates = {'normal': RSNA_CLASS_PROMPTS_webdataset['Normal'],
                                          'pneumonia': temp['Pneumonia']}
                else:
                    k = 1  # This means all 10 templates to be used...
                    print(f"Using {k - 1} templates for the ensembling at test time")
                    for single_key in templates.keys():
                        templates[single_key] = templates[single_key][0:k]
                    prompted_templates = templates
                text_features, mean, std = build_text_features(prompted_templates, None, model, tokenizer,
                                                               classnorm=classnorm)
            else:
                if type(templates) == dict:
                    k = 11  # This means all 10 templates to be used...
                    print(f"Using {k - 1} templates for the ensembling at test time")
                    for single_key in templates.keys():
                        length = len(templates[single_key])
                        templates[single_key] = templates[single_key][0:length]
                    prompted_templates = templates
                else:
                    prompted_templates = templates
                text_features, mean, std = build_text_features(prompted_templates, labels, model, tokenizer,
                                                               classnorm=classnorm)
        if isinstance(samples, tuple) or isinstance(samples, list):
            images, target = samples[0], samples[1]
        elif isinstance(samples, dict):
            images, target = samples["pixel_values"], samples["targets"]
        else:
            raise ValueError("unknown sample type", type(samples))

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # encode images
        image_features = model.encode_image(images)

        if classnorm:
            image_features = (image_features - mean) / std

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logits_per_image = image_features @ text_features.t()
        logits_per_image = logits_per_image.cpu()
        target = target.cpu()
        if name == "chexpert-5x200":
            # convert to label encoding
            target = torch.argmax(target, axis=1)
        if is_acc:
            # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            total_top1 += correct.item()
            total_images += images.size(0)
            if name == "radimagenet":
                # Update per-class accuracy counts
                for t, p in zip(target, pred):
                    class_correct[t.item()] += p.eq(t).item()
                    class_total[t.item()] += 1
            # Also save those to have results for the other metrics
            all_outputs.append(logits_per_image)
            all_targets.append(target)
        else:
            all_outputs.append(logits_per_image)
            all_targets.append(target)

    if is_acc:
        if name == "radimagenet":
            # Now calculate accuracies for each modality
            US_all_class_correct = 0
            MRI_all_class_correct = 0
            CT_all_class_correct = 0
            US_all_class_total = 0
            MRI_all_class_total = 0
            CT_all_class_total = 0
            for single_us_index in modality_indices_radimagenet_test_set['US']:
                US_all_class_correct += class_correct[single_us_index]
                US_all_class_total += class_total[single_us_index]
            for single_mri_index in modality_indices_radimagenet_test_set['MRI']:
                MRI_all_class_correct += class_correct[single_mri_index]
                MRI_all_class_total += class_total[single_mri_index]
            for single_ct_index in modality_indices_radimagenet_test_set['CT']:
                CT_all_class_correct += class_correct[single_ct_index]
                CT_all_class_total += class_total[single_ct_index]

            return 100 * total_top1 / total_images, \
                   100 * US_all_class_correct / US_all_class_total, \
                   100 * MRI_all_class_correct / MRI_all_class_total, \
                   100 * CT_all_class_correct / CT_all_class_total
        if name == 'radimagenet' or name == 'chexpert-5x200' or name == 'CT_sagittal' or name == 'CT_axial' \
                or name == 'CT_coronal' or name == 'dr_uwf' or name == 'dr_regular' \
                or name == 'PCAM' or name == 'LC25000_lung' or name == 'LC25000_colon' \
                or name == "NCK_CRC" or name == 'BACH' or name == 'Osteo' \
                or name == 'skin_cancer' or name == 'skin_tumor' or name == 'SICAPv2' \
                or name == 'five_retina' or name == 'odir_retina':
            return 100 * total_top1 / total_images
        else:
            # Now also return the other metric results
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)
            accuracy = 100 * total_top1 / total_images
            auc_roc = roc_auc(all_outputs, all_targets)
            f1_score = F1_score(all_outputs, all_targets)
            precision_score = Precision_score(all_outputs, all_targets)
            recall_score = Recall_score(all_outputs, all_targets)
            return {"acc": accuracy, "auc_roc": auc_roc, "f1_score": f1_score,
                    "precision_score": precision_score, "recall_score": recall_score}
    else:
        return torch.cat(all_outputs), torch.cat(all_targets)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def Recall_score(outputs, targets):
    pred = outputs.argmax(1)
    Recall_score = metrics.recall_score(targets, pred)
    return 100 * Recall_score


def F1_score(outputs, targets):
    pred = outputs.argmax(1)
    F1_score = metrics.f1_score(targets, pred)
    return 100 * F1_score


def Precision_score(outputs, targets):
    pred = outputs.argmax(1)
    Precision_score = metrics.precision_score(targets, pred)
    return 100 * Precision_score


def mean_per_class(outputs, targets):
    pred = outputs.argmax(1)
    confusion_matrix = metrics.confusion_matrix(targets, pred)
    per_classes = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)

    return 100 * per_classes.mean()


def roc_auc(outputs, targets):
    pos_score = outputs[:, 1] - outputs[:, 0]
    metric = metrics.roc_auc_score(targets, pos_score)

    return 100 * metric


if __name__ == '__main__':
    logits = torch.randn(128, 10)
    targets = torch.randint(size=(128,), low=0, high=10)

    evaluate("imagenet", logits, targets)

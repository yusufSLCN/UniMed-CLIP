# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import json
import os
import pickle
import zipfile
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import ast
import torch
import random
from constants import CHEXPERT_COMPETITION_TASKS
from torchvision import transforms
from torchvision import datasets as t_datasets
import ast

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
                                         transform=transform)
    elif entry['type'] == 'special':
        if name == 'CIFAR10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                                         transform=transform, download=True)
        elif name == 'CIFAR100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                                          transform=transform, download=True)
        elif name == 'STL10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                                       transform=transform, download=True)
        elif name == 'MNIST':
            dataset = t_datasets.MNIST(root, train=is_train,
                                       transform=transform, download=True)
        elif name == 'chexpert-5x200':
            dataset = ZeroShotImageDataset(['chexpert-5x200'], CHEXPERT_COMPETITION_TASKS, transform
                                           , parent_data_path=root)
        elif name == "radimagenet":
            dataset = RadImageNet(root, transform)
        elif name == "rsna_pneumonia":
            dataset = RSNA_Pneumonia(root, transform)
        elif name == "thyroid_us":
            dataset = thyroid_us_and_breast(root, transform, "thyroid_test_fold1.csv")
        elif name == "breast_us":
            dataset = thyroid_us_and_breast(root, transform, "breast_test_fold1.csv")
        elif name == "meniscal_mri":
            dataset = meniscal_mri(root, transform, "meniscus_test_fold1.csv")
        elif name == 'acl_mri':
            dataset = acl_mri(root, transform, "test_fold1.csv")
        elif name == 'CT_axial':
            dataset = CT_dataset(root, transform, "organs_axial")
        elif name == 'CT_coronal':
            dataset = CT_dataset(root, transform, "organs_coronal")
        elif name == 'CT_sagittal':
            dataset = CT_dataset(root, transform, "organs_sagittal")
        elif name == 'dr_regular':
            dataset = CT_dataset(root, transform, "dr_regular")
        elif name == 'dr_uwf':
            dataset = CT_dataset(root, transform, "dr_uwf")
        elif name == 'LC25000_lung':
            dataset = LC25000(root, transform, "lung")
        elif name == 'LC25000_colon':
            dataset = LC25000(root, transform, "colon")
        elif name == 'PCAM':
            dataset = PCAM(root, transform, "PCam_Test_preprocessed")
        elif name == 'NCK_CRC':
            dataset = NCK_CRC(root, transform, "CRC-VAL-HE-7K")
        elif name == 'BACH':
            dataset = BACH(root, transform, "BACH")
        elif name == 'Osteo':
            dataset = Osteo(root, transform, "Osteosarcoma")
        elif name == 'skin_cancer':
            dataset = Skin_datasets(root, transform, "skin_tumor", 'cancer')
        elif name == 'skin_tumor':
            dataset = Skin_datasets(root, transform, "skin_tumor", 'tumor')
        elif name == 'refuge_retina':
            dataset = Retina_datasets(root, transform, '25_REFUGE.csv')
        elif name == 'five_retina':
            dataset = Retina_datasets(root, transform, '13_FIVES.csv')
        elif name == 'odir_retina':
            dataset = Retina_datasets(root, transform, '08_ODIR200x3.csv')

    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'CLEVRCounts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8',
                                          'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception('Unknown dataset')

    return dataset


class ZeroShotImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 datalist=['chexpert-5x200'],
                 class_names=None,
                 imgtransform=None,
                 parent_data_path="",
                 ) -> None:
        '''support data list in mimic-5x200, chexpert-5x200, rsna-balanced-test, covid-test
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        self.transform = imgtransform

        self.class_names = class_names
        self.parent_data_path = parent_data_path
        # imgpath, subject_id, report, labels...(14 labels)
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}.csv'
            print('load data from', filename)
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.parent_data_path, row.imgpath))
        # img = self._pad_img(img)
        img = self.transform(img)
        label = torch.from_numpy(row[self.class_names].values.astype(np.float_))
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        '''pad img to square.
        '''
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)


class RadImageNet(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(os.path.join(parent_path, "radimagenet_test_set_formatted.csv"))
        self.transform = transform
        self.parent_data_path = parent_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        img_name = ast.literal_eval(img_name)[0]
        img = Image.open(os.path.join(self.parent_data_path, img_name))
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(img)

        return image, label


class CT_dataset(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, foldername=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        all_data = pd.read_csv(os.path.join(os.path.join(parent_path, foldername),
                                            "annotations.csv"))
        # Filter the data to only retain the test samples
        self.data = all_data[all_data['split'] == 'test']
        self.transform = transform
        self.parent_data_path = os.path.join(parent_path, foldername)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        img = Image.open(os.path.join(self.parent_data_path, img_name))
        label = self.data.iloc[idx, 2]

        if self.transform:
            image = self.transform(img)

        return image, label


class LC25000(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, split=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if split == "lung":
            classes = ['lung_aca', 'lung_n', 'lung_scc']
        else:
            classes = ['colon_aca', 'colon_n']
        self.split = split
        self.class_name_folder = []
        self.images = []
        self.labels = []
        for idx, single_class_folder in enumerate(classes):
            images_per_classes = list(os.listdir(os.path.join(parent_path, single_class_folder)))
            self.images = self.images + images_per_classes
            self.labels = self.labels + ([idx] * len(images_per_classes))
            self.class_name_folder = self.class_name_folder + ([single_class_folder] * len(images_per_classes))
        self.transform = transform
        self.parent_data_path = parent_path
        assert len(self.images) == len(self.labels) == len(self.class_name_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        class_folder_name = self.class_name_folder[idx]
        img = Image.open(os.path.join(os.path.join(self.parent_data_path, class_folder_name), img_name))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(img)

        return image, label


class PCAM(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, foldername=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        all_files = os.listdir(os.path.join(parent_path, foldername))
        # Filter the data to only retain the test samples
        self.data = all_files
        # Create labels
        labels = []
        for single_file in all_files:
            splitted_label = int(single_file.split("_")[1].split(".")[0])
            labels.append(splitted_label)
        self.labels = labels
        self.transform = transform
        self.parent_data_path = os.path.join(parent_path, foldername)
        assert len(self.labels) == len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data[idx]
        img = Image.open(os.path.join(self.parent_data_path, img_name))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(img)

        return image, label


class NCK_CRC(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, foldername=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        NCK_CRC_converter = {"ADI": 0,
                             "DEB": 1,
                             "LYM": 2,
                             "MUC": 3,
                             "MUS": 4,
                             "NORM": 5,
                             "STR": 6,
                             "TUM": 7,
                             }
        all_data = []
        all_class_names = []
        all_labels = []
        folder_names = os.listdir(os.path.join(parent_path, foldername))
        for single_folder in folder_names:
            class_path = os.path.join(os.path.join(parent_path, foldername), single_folder)
            images_inside_folder = os.listdir(class_path)
            class_label = [NCK_CRC_converter[single_folder]] * len(images_inside_folder)
            all_data.extend(images_inside_folder)
            all_labels.extend(class_label)
            all_class_names.extend([single_folder] * len(images_inside_folder))
        # Filter the data to only retain the test samples
        self.data = all_data
        self.labels = all_labels
        self.prefix_name = all_class_names
        assert len(self.data) == len(self.labels) == len(self.prefix_name)
        self.transform = transform
        self.parent_data_path = os.path.join(parent_path, foldername)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data[idx]
        class_name = self.prefix_name[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(self.parent_data_path, os.path.join(class_name, img_name)))
        if self.transform:
            image = self.transform(img)

        return image, label


class BACH(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, foldername=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        all_data = pd.read_csv(os.path.join(os.path.join(parent_path, foldername),
                                            "microscopy_ground_truth.csv"))

        self.data = all_data
        self.transform = transform
        self.parent_data_path = os.path.join(parent_path, foldername)
        self.label_to_text_mapping = {'Normal': 3, 'Invasive': 2, 'InSitu': 1, "Benign": 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        label_text = self.data.iloc[idx, 1]
        img = Image.open(os.path.join(self.parent_data_path, label_text + "/" + img_name))

        label = self.label_to_text_mapping[label_text]

        if self.transform:
            image = self.transform(img)

        return image, label


class Retina_datasets(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, data=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        filename = f'./local_data/{data}'
        all_data = pd.read_csv(filename)

        self.data = all_data
        self.transform = transform
        self.parent_data_path = parent_path

        if data == '25_REFUGE.csv':
            self.label_to_text_mapping = {'no glaucoma': 0, 'glaucoma': 1}
        elif data == '13_FIVES.csv':
            self.label_to_text_mapping = {"age related macular degeneration": 0,
                                          "diabetic retinopathy": 1,
                                          "glaucoma": 2,
                                          "normal": 3}
        elif data == '08_ODIR200x3.csv':
            self.label_to_text_mapping = {"normal": 0,
                                          "pathologic myopia": 1,
                                          "cataract": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 1]
        label_text = self.data.iloc[idx, 3]  # it is a list
        label_text = ast.literal_eval(label_text)[0]
        img = Image.open(os.path.join(self.parent_data_path, img_name))

        label = self.label_to_text_mapping[label_text]

        if self.transform:
            image = self.transform(img)

        return image, label

class Skin_datasets(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, foldername=None, split_type='cancer'):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if split_type == 'cancer':
            all_data = pd.read_csv(os.path.join(os.path.join(parent_path, foldername),
                                                "data/tiles-v2.csv"))
            # Filter the dataset and take only test samples...
            all_data = all_data[all_data['set'] == 'Test']
            self.label_to_text_mapping = {
                "nontumor_skin_necrosis_necrosis": 0,

                "nontumor_skin_muscle_skeletal":
                    1,

                "nontumor_skin_sweatglands_sweatglands":
                    2,

                "nontumor_skin_vessel_vessel":
                    3,

                "nontumor_skin_elastosis_elastosis":
                    4,

                "nontumor_skin_chondraltissue_chondraltissue":
                    5,

                "nontumor_skin_hairfollicle_hairfollicle":
                    6,
                "nontumor_skin_epidermis_epidermis": 7,
                "nontumor_skin_nerves_nerves":
                    8,

                "nontumor_skin_subcutis_subcutis":
                    9,

                "nontumor_skin_dermis_dermis":
                    10,

                "nontumor_skin_sebaceousglands_sebaceousglands":
                    11,

                "tumor_skin_epithelial_sqcc":
                    12,

                "tumor_skin_melanoma_melanoma":
                    13,

                "tumor_skin_epithelial_bcc":
                    14,

                "tumor_skin_naevus_naevus":
                    15
            }
        else:
            all_data = pd.read_csv(os.path.join(os.path.join(parent_path, foldername),
                                                "data/SkinTumorSubset.csv"))
            # Filter the dataset and take only test samples...
            all_data = all_data[all_data['set'] == 'Test']
            self.label_to_text_mapping = {"tumor_skin_epithelial_sqcc":
                                              0,

                                          "tumor_skin_melanoma_melanoma":
                                              1,

                                          "tumor_skin_epithelial_bcc":
                                              2,

                                          "tumor_skin_naevus_naevus":
                                              3
                                          }

        self.data = all_data
        self.transform = transform
        self.parent_data_path = os.path.join(parent_path, foldername)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 1]
        label_text = self.data.iloc[idx, 2]
        img = Image.open(os.path.join(self.parent_data_path, img_name))

        label = self.label_to_text_mapping[label_text]

        if self.transform:
            image = self.transform(img)

        return image, label


class Osteo(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None, foldername=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        all_data = pd.read_csv(os.path.join(os.path.join(parent_path, foldername),
                                            "annotations_final.csv"))

        self.data = all_data
        self.transform = transform
        self.parent_data_path = os.path.join(parent_path, foldername)
        self.label_to_text_mapping = {'Viable': 2, 'Non-Tumor': 0, "Non-Viable-Tumor": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        label_text = self.data.iloc[idx, 1]
        img = Image.open(os.path.join(self.parent_data_path, img_name))

        label = self.label_to_text_mapping[label_text]

        if self.transform:
            image = self.transform(img)

        return image, label


class RSNA_Pneumonia(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(os.path.join(parent_path, "RSNA_pneumonia_balanced_testfile.csv"))
        self.transform = transform
        self.parent_data_path = parent_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 1]
        img = Image.open(os.path.join(self.parent_data_path, img_name))
        label = self.data.iloc[idx, 2]

        if self.transform:
            image = self.transform(img)

        return image, label


class thyroid_us_and_breast(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform, csv_file_name):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(os.path.join(parent_path, csv_file_name))
        self.transform = transform
        self.parent_data_path = parent_path
        # self.mapping = {'malignant': 0, 'benign': 1}
        self.mapping = {'malignant': 1, 'benign': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        img = Image.open(os.path.join(self.parent_data_path, img_name))
        label_name = self.data.iloc[idx, 1]
        label = self.mapping[label_name]

        if self.transform:
            image = self.transform(img)

        return image, label


class meniscal_mri(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform, csv_file_name):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(os.path.join(parent_path, csv_file_name))
        self.transform = transform
        self.parent_data_path = parent_path
        self.mapping = {'p': 1, 'n': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        img = Image.open(os.path.join(self.parent_data_path, img_name))
        label_name = self.data.iloc[idx, 1]
        label = self.mapping[label_name]

        if self.transform:
            image = self.transform(img)

        return image, label


class acl_mri(torch.utils.data.Dataset):
    def __init__(self, parent_path, transform, csv_file_name):
        """
        Args:
            csv_file (string): Path to the CSV file containing image paths and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(os.path.join(parent_path, csv_file_name))
        self.transform = transform
        self.parent_data_path = parent_path
        self.mapping = {'yes': 1, 'no': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx, 0]
        img = Image.open(os.path.join(self.parent_data_path, img_name))
        label_name = self.data.iloc[idx, 1]
        label = self.mapping[label_name]

        if self.transform:
            image = self.transform(img)

        return image, label

import json
import os
import tarfile
import io
import argparse
from tqdm import tqdm

def create_webdataset(json_file_path, output_dir, parent_dataset_path, tar_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_file_path, "r") as f:
        json_dict = json.load(f)
        tar_index = 0
        file_count = 0
        tar = None
        # One for loop for main caption
        for single_key in json_dict.keys():

            my_list = json_dict[single_key]  # this is a list
            for single_entry in tqdm(my_list):
                # Read the image file
                filename = single_entry["pair_id"] + ".jpg"
                image_path = os.path.join(parent_dataset_path, filename)
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                except:
                    print(f"image not found: {image_path}, skipping... ")
                    continue

                # label = ast.literal_eval(row[1])
                all_caption = single_entry["fig_caption"] # GPT response...
                if str(all_caption) == 'nan':
                    print(f"original caption not found: {image_path}, skipping... ")
                    continue
                caption = all_caption.strip().strip("\n\n").strip("\n")
                if file_count % tar_size == 0:
                    if tar:
                        tar.close()
                    tar_index += 1
                    tar_path = os.path.join(output_dir, f"dataset-{tar_index:06d}.tar")
                    tar = tarfile.open(tar_path, 'w')

                # Create an in-memory tarfile
                img_tarinfo = tarfile.TarInfo(name=f"{file_count:06d}.jpg")
                img_tarinfo.size = len(img_data)
                tar.addfile(img_tarinfo, io.BytesIO(img_data))

                # Add caption.txt to the tarfile
                caption_data = caption.encode('utf-8')
                caption_tarinfo = tarfile.TarInfo(name=f"{file_count:06d}.txt")
                caption_tarinfo.size = len(caption_data)
                tar.addfile(caption_tarinfo, io.BytesIO(caption_data))

                file_count += 1

        # One for loop for inline mention as the captions...
        for single_key in json_dict.keys():

            my_list = json_dict[single_key]  # this is a list
            for single_entry in tqdm(my_list):
                # Read the image file
                filename = single_entry["pair_id"] + ".jpg"
                image_path = os.path.join(parent_dataset_path, filename)
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                except:
                    print(f"image not found: {image_path}, skipping... ")
                    continue

                if single_entry["in_text_mention"] is None:
                    print(f"Inline caption not found: {image_path}, skipping... ")
                    continue
                all_caption = single_entry["in_text_mention"][0]['tokens'] # GPT response...
                caption = all_caption.strip().strip("\n\n").strip("\n")
                if file_count % tar_size == 0:
                    if tar:
                        tar.close()
                    tar_index += 1
                    tar_path = os.path.join(output_dir, f"dataset-{tar_index:06d}.tar")
                    tar = tarfile.open(tar_path, 'w')

                # Create an in-memory tarfile
                img_tarinfo = tarfile.TarInfo(name=f"{file_count:06d}.jpg")
                img_tarinfo.size = len(img_data)
                tar.addfile(img_tarinfo, io.BytesIO(img_data))

                # Add caption.txt to the tarfile
                caption_data = caption.encode('utf-8')
                caption_tarinfo = tarfile.TarInfo(name=f"{file_count:06d}.txt")
                caption_tarinfo.size = len(caption_data)
                tar.addfile(caption_tarinfo, io.BytesIO(caption_data))

                file_count += 1
        if tar:
            tar.close()


# Usage example
json_file = '/home/muzammal/uzair_experiments/datasets/llava_med/llava_med_instruct_fig_captions.json'
output_dir = '/home/muzammal/uzair_experiments/datasets/llava_med/llava_med_hq_60k_set_webdataset/'
parent_dataset_path = '/home/muzammal/uzair_experiments/datasets/llava_med/images/'
create_webdataset(json_file, output_dir, parent_dataset_path)
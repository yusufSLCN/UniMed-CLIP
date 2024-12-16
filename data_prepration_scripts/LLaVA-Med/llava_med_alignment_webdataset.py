import json
import os
import tarfile
import io
from tqdm import tqdm
import argparse

def create_webdataset(json_file_path, output_dir, parent_dataset_path, tar_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_file_path, "r") as f:
        json_dict = json.load(f)
        tar_index = 0
        file_count = 0
        tar = None

        for single_dict in tqdm(json_dict):
            row = single_dict
            # Read the image file
            filename = row["image"]
            image_path = os.path.join(parent_dataset_path, filename)
            try:
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
            except:
                print(f"image not found: {image_path}, skipping... ")
                continue
            if file_count % tar_size == 0:
                if tar:
                    tar.close()
                tar_index += 1
                tar_path = os.path.join(output_dir, f"dataset-{tar_index:06d}.tar")
                tar = tarfile.open(tar_path, 'w')


            # label = ast.literal_eval(row[1])
            all_caption = row["conversations"][1]["value"]  # GPT response...
            caption = all_caption.strip().strip("\n\n").strip("\n")

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

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Create a WebDataset from CSV")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to the CSV file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store the output tar files")
    parser.add_argument('--parent_dataset_path', type=str, required=True,
                        help="Path to the parent dataset containing images")
    parser.add_argument('--tar_size', type=int, default=1000, help="Number of files per tar file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    create_webdataset(args.csv_file, args.output_dir, args.parent_dataset_path, args.tar_size)

# Usage example
json_file = '/home/muzammal/uzair_experiments/datasets/llava_med/llava_med_alignment_500k_filtered.json'
output_dir = '/home/muzammal/uzair_experiments/datasets/llava_med/llava_med_alignment_set_webdataset/'
parent_dataset_path = '/home/muzammal/uzair_experiments/datasets/llava_med/images/'
create_webdataset(json_file, output_dir, parent_dataset_path)
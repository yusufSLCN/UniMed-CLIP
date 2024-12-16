import os
import tarfile
import io
import json
from tqdm.contrib import tzip
import argparse

def create_webdataset(orig_file_path, refined_file_path, output_dir, parent_dataset_path, tar_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    with open(orig_file_path, "r") as f:
        reader = json.load(f)['annotations']
        reader_refined = json.load(open(refined_file_path, "r"))['refined_captions']
        tar_index = 0
        file_count = 0
        tar = None

        for full_caption, refined_caption_list in tzip(reader, reader_refined):
            if file_count % tar_size == 0:
                if tar:
                    tar.close()
                tar_index += 1
                tar_path = os.path.join(output_dir, f"dataset-{tar_index:06d}.tar")
                tar = tarfile.open(tar_path, 'w')

            filename = full_caption['image_id']
            caption_orig = [full_caption['caption']]
            caption_orig += refined_caption_list

            all_caption = caption_orig
            caption = ''
            for single_caption in all_caption: caption += single_caption.strip('.') + "._openi_"
            # Read the image file
            image_path = os.path.join(parent_dataset_path, filename + ".png")
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()

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
    parser.add_argument('--original_json_file_summarizations_path', type=str, required=True)
    parser.add_argument('--gpt_text_descriptions_path', type=str, required=True, help="Path to the CSV file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store the output tar files")
    parser.add_argument('--parent_dataset_path', type=str, required=True,
                        help="Path to the parent dataset containing images")
    parser.add_argument('--tar_size', type=int, default=1000, help="Number of files per tar file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    create_webdataset(args.original_json_file_summarizations_path, args.gpt_text_descriptions_path, args.output_dir,
                      args.parent_dataset_path)

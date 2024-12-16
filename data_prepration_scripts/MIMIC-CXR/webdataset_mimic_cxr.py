import os
import tarfile
import io
import pandas as pd
import ast
from tqdm import tqdm
import argparse

def create_webdataset(csv_file, output_dir, parent_dataset_path, tar_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, newline='') as f:
        reader = pd.read_csv(csv_file, delimiter=',')

        tar_index = 0
        file_count = 0
        tar = None
        total_skipped = 0
        for row in tqdm(reader.values):
            filename = ast.literal_eval(row[0])[0]
            # Read the image file
            image_path = os.path.join(parent_dataset_path, filename)
            if not os.path.exists(image_path):
                print(f"Image path {image_path} does not exist")
                print(f"Skipping this iteration")
                total_skipped += 1
                continue
            if file_count % tar_size == 0:
                if tar:
                    tar.close()
                tar_index += 1
                tar_path = os.path.join(output_dir, f"dataset-{tar_index:06d}.tar")
                tar = tarfile.open(tar_path, 'w')

            label = ast.literal_eval(row[1])
            all_caption = ast.literal_eval(row[2])
            report = row[3]
            if type(report) != str:
                report = "noreportpresent"
            caption = ''
            # print(f"REPORT IS {report} \n")
            for single_caption in all_caption: caption += single_caption.strip('.') + "._mimiccxr_"
            caption = caption + "_mimiccxr_" + report
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()

            # Create an in-memory tarfile
            img_tarinfo = tarfile.TarInfo(name=f"{file_count:06d}.jpg")
            img_tarinfo.size = len(img_data)
            tar.addfile(img_tarinfo, io.BytesIO(img_data))

            # Add label.txt to the tarfile
            label_data = label[0].encode('utf-8')
            label_tarinfo = tarfile.TarInfo(name=f"{file_count:06d}.cls")
            label_tarinfo.size = len(label_data)

            # Add caption.txt to the tarfile
            caption_data = caption.encode('utf-8')
            caption_tarinfo = tarfile.TarInfo(name=f"{file_count:06d}.txt")
            caption_tarinfo.size = len(caption_data)
            tar.addfile(caption_tarinfo, io.BytesIO(caption_data))

            file_count += 1

        if tar:
            tar.close()

        print(f"Total {total_skipped} files have been skipped because no image was found for that in our folder.")

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

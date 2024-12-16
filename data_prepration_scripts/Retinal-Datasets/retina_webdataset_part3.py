import os
import tarfile
import io
import pandas as pd
from tqdm import tqdm
import argparse

banned_categories = ['myopia', 'cataract', 'macular hole', 'retinitis pigmentosa', "myopic", "myope", "myop", "retinitis"]

def create_webdataset(main_csv_directory, image_dir_path, output_dir, tar_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    # Load both csv files
    tar_index = 0
    file_count = 0
    tar = None
    # Now lets do it for that vision language model
    dataframe = pd.read_csv(main_csv_directory + "39_MM_Retinal_dataset.csv")
    selected_id_list = range(len(dataframe))  # 100%数据   100% data

    for i in tqdm(selected_id_list):
        if file_count % tar_size == 0:
            if tar:
                tar.close()
            tar_index += 1
            tar_path = os.path.join(output_dir, f"dataset-{tar_index:06d}.tar")
            tar = tarfile.open(tar_path, 'w')
        sample_df = dataframe.loc[i, :].to_dict()  # image,attributes,categories   Turn each line into a dictionary
        image_path = sample_df["image"]
        caption = sample_df["caption"]
        final_caption = caption
        image_file_name = image_path
        # Now need to process the captions
        if str(final_caption) == 'nan':
            continue

        caption = final_caption
        # Read the image file
        image_path = os.path.join(image_dir_path, image_file_name)
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
        except:
            print(f"image not found: {image_path} \n subset is {image_file_name} ")
            continue

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
    parser.add_argument('--csv_files_directory', type=str, required=True, help="Path to the CSV files for all datasets")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store the output tar files")
    parser.add_argument('--parent_datasets_path', type=str, required=True,
                        help="Path to the parent folder containing Retina Datasets folders")
    parser.add_argument('--tar_size', type=int, default=1000, help="Number of files per tar file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    create_webdataset(args.csv_file, args.output_dir, args.parent_dataset_path, args.tar_size)

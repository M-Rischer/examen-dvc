import requests
import os
import logging
from check_structure import check_existing_file, check_existing_folder

def import_raw_data(raw_data_relative_path, 
                    filenames,
                    bucket_folder_url):
    '''Import filenames from bucket_folder_url into raw_data_relative_path'''

    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path, exist_ok=True)

    for filename in filenames:
        input_file = os.path.join(bucket_folder_url, filename)
        output_file = os.path.join(raw_data_relative_path, filename)

        if check_existing_file(output_file):
            print(f'Downloading {input_file} as {os.path.basename(output_file)}...')
            response = requests.get(input_file)
            if response.status_code == 200:
                with open(output_file, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Error downloading {input_file} — Status code: {response.status_code}")
        else:
            print(f"File {output_file} already exists, skipping download.")

def main(raw_data_relative_path="./data/raw_data", 
         filenames=["raw.csv"],
         bucket_folder_url="https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr"):
    """Download flotation dataset into ./data/raw_data"""
    import_raw_data(raw_data_relative_path, filenames, bucket_folder_url)
    logger = logging.getLogger(__name__)
    logger.info("Raw data has been downloaded successfully.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()

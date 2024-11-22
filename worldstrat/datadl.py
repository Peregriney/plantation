########################################################
# SET THE VARIABLE BELOW TO TRUE TO DOWNLOAD THE DATASET
DOWNLOAD = True
########################################################

from src.utils.dataset_download_tools import *
os.makedirs('dataset_download', exist_ok=True)
files_to_download = {
    './dataset_download/hr_dataset_raw.tar.gz' : 'https://zenodo.org/record/6810792/files/hr_dataset_raw.tar.gz?download=1',
    './dataset_download/lr_dataset_l2a.tar.gz' : 'https://zenodo.org/record/6810792/files/lr_dataset_l2a.tar.gz?download=1',
    './dataset_download/lr_dataset_l1c.tar.gz' : 'https://zenodo.org/record/6810792/files/lr_dataset_l1c.tar.gz?download=1',
    './dataset_download/metadata.csv' : 'https://zenodo.org/record/6810792/files/metadata.csv?download=1',
    './dataset_download/stratified_train_val_test_split.csv' : 'https://zenodo.org/record/6810792/files/stratified_train_val_test_split.csv?download=1',
}
if DOWNLOAD:
    for file, link in files_to_download.items():
        download_large_file(link, file)

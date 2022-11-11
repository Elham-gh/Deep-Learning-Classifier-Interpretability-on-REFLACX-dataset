import os

h5_path = '/scratch/elham/rawdata' #Location where hdf5 files with preprocessed datasets will be saved

jpg_path = '/usr/sci/scratch/ricbl/mimic-cxr/mimic-cxr-jpg/mimic-cxr-jpg-2.0.0.physionet.org/'  # Location of the files folder of the MIMIC-CXR dataset
mimic_dir = '/usr/sci/scratch/ricbl/mimic-cxr/' # Location of the tables mimic-cxr-2.0.0-chexpert.csv and mimic-cxr-2.0.0-split.csv from the MIMIC-CXR-JPG dataset
path_chexpert_labels = './' #location of the files containing the labels for the reports of the REFLACX dataset when using the modified chexpert-labeler
metadata_et_location = '/usr/sci/scratch/ricbl/reflacx/main_data/' # location of the metadata tables of the REFLACX dataset
eyetracking_dataset_path = '/usr/sci/scratch/ricbl/reflacx/main_data/' # location of the main content of the REFLACX dataset
preprocessed_heatmaps_location = '/home/sci/elham/datasets/REFLACX_Full' # location where the folders containing precalculated heatmaps are saved
# preprocessed_heatmaps_location = '/home/sci/elham/datasets/REFLACX_Full/one_hm_p_image_3' # location where the folders containing precalculated heatmaps are saved
# Z:\datasets\REFLACX_Full\heatmaps_sentences_phase_3 # image_index/0/sentence_number
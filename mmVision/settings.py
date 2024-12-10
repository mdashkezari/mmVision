"""
Author: Mohammad Dehghani Ashkezari <mdehghan@uw.edu>

Function: Holds cross-project settings and constants.
"""

import os



########### project directories ###########
HOME = f"{os.path.realpath(os.path.dirname(__file__))}/../"
FIG_DIR = f"{HOME}fig/"                                            # path to the saved figures directory
MODEL_DIR = f"{HOME}model/"                                        # path to the saved models directory
TENSORBOARD_DIR = f"{HOME}tensorboard/"                            # path to the tensorboard directory
CHECKPOINT_DIR = f"{HOME}checkpoint/"                              # path to the checkpoint directory
INFERENCE_DIR = f"{HOME}inference/"                                

# DATA_DIR = f"{HOME}data/"                                          # path to the data directory where the original raw data and processed versions are stored.
DATA_DIR = f"/data/dev/microscope/data/"                                          # path to the data directory where the original raw data and processed versions are stored.

TRAIN_TEST_DIR = f"{HOME}data/traintest/"                          # path to dir where different versions of train-test datasets are stored
#################################################

################# BERINGSEA (ZOO) #################
BERING_RAW_DATA_DIR = f"{DATA_DIR}source/BeringSea/8146283/Dataset_BeringSea/Dataset_BeringSea/"      # path to the original raw dataset

BERING_TRAIN_DIR_ALL = f"{BERING_RAW_DATA_DIR}train/"                  
BERING_TEST_DIR_ALL = f"{BERING_RAW_DATA_DIR}test/"                  
#################################################

################# ZOOSCAN  #################
ZOO_RAW_DATA_DIR = f"{DATA_DIR}source/ZooScanExample/ZooScanSet/"          # path to the original raw dataset
ZOO_RAW_IMG_DIR = f"{ZOO_RAW_DATA_DIR}imgs/"                               # path to the original raw images
ZOO_BALANCED_1000_DIR = f"{ZOO_RAW_DATA_DIR}balanced_1000/" 

ZOO_TRAIN_TEST_DIR_ALL = f"{TRAIN_TEST_DIR}zooscan/all/"                   # path to dir where 100% of the original dataset (raw) is split into train-test
ZOO_TRAIN_DIR_ALL = f"{ZOO_TRAIN_TEST_DIR_ALL}train/"                  
ZOO_TEST_DIR_ALL = f"{ZOO_TRAIN_TEST_DIR_ALL}test/"                  

ZOO_TRAIN_TEST_DIR_10PCT = f"{TRAIN_TEST_DIR}zooscan/10pct/"               # path to dir where 10% of the full tarining dataset (TRAIN_TEST_DIR_ALL) is used. The test dataset remains the same as TRAIN_TEST_DIR_ALL.
ZOO_TRAIN_DIR_10PCT = f"{ZOO_TRAIN_TEST_DIR_10PCT}train/"                  
ZOO_TEST_DIR_10PCT = f"{ZOO_TRAIN_TEST_DIR_10PCT}test/"                  

ZOO_TRAIN_TEST_DIR_BALANCED_1000 = f"{TRAIN_TEST_DIR}zooscan/balanced_1000/"                   
ZOO_TRAIN_DIR_BALANCED_1000 = f"{ZOO_TRAIN_TEST_DIR_BALANCED_1000}train/"                  
ZOO_VALID_DIR_BALANCED_1000 = f"{ZOO_TRAIN_TEST_DIR_BALANCED_1000}valid/"                  
ZOO_TEST_DIR_BALANCED_1000 = f"{ZOO_TRAIN_TEST_DIR_BALANCED_1000}test/"                  
#################################################

################# BERING (ZOO) #################
ZOO_BERING_RAW_DATA_DIR = f"{DATA_DIR}source/ZooVis_BeringSea/all/Dataset_BeringSea/Dataset_BeringSea/"      # path to the original raw dataset
ZOO_BERING_BALANCED_2000_DIR = f"{DATA_DIR}source/ZooVis_BeringSea/balanced_2000/" 

ZOO_BERING_TRAIN_TEST_DIR_BALANCED_2000 = f"{TRAIN_TEST_DIR}bering/balanced_2000/"                   
ZOO_BERING_TRAIN_DIR_BALANCED_2000 = f"{ZOO_BERING_TRAIN_TEST_DIR_BALANCED_2000}train/"                  
ZOO_BERING_VALID_DIR_BALANCED_2000 = f"{ZOO_BERING_TRAIN_TEST_DIR_BALANCED_2000}valid/"                  
ZOO_BERING_TEST_DIR_BALANCED_2000 = f"{ZOO_BERING_TRAIN_TEST_DIR_BALANCED_2000}test/"                  
#################################################

################# ISIIS COWEN #################
ISIIS_COWEN_RAW_DATA_DIR = f"{DATA_DIR}source/ISIIS_Cowen/0127422.2.3/0127422/2.3/data/0-data/FINAL_Plankton_Segments_12082014/"      # path to the original raw dataset
ISIIS_COWEN_BALANCED_1000_DIR = f"{DATA_DIR}source/ISIIS_Cowen/balanced_1000/" 

ISIIS_COWEN_TRAIN_TEST_DIR_BALANCED_1000 = f"{TRAIN_TEST_DIR}isiis_cowen/balanced_1000/"                   
ISIIS_COWEN_TRAIN_DIR_BALANCED_1000 = f"{ISIIS_COWEN_TRAIN_TEST_DIR_BALANCED_1000}train/"                  
ISIIS_COWEN_VALID_DIR_BALANCED_1000 = f"{ISIIS_COWEN_TRAIN_TEST_DIR_BALANCED_1000}valid/"                  
ISIIS_COWEN_TEST_DIR_BALANCED_1000 = f"{ISIIS_COWEN_TRAIN_TEST_DIR_BALANCED_1000}test/"                  
#################################################

################# IFCB MAINE #################
IFCB_RAW_DIR = f"{DATA_DIR}source/IFCB/classes/"  
IFCB_BALANCED_1000_DIR = f"{DATA_DIR}source/IFCB/balanced_1000/"  

IFCB_TRAIN_TEST_DIR_ALL = f"{TRAIN_TEST_DIR}ifcb/all/"                       # path to dir where 100% of the original ifcb dataset (raw) is split into train-test
IFCB_TRAIN_DIR_ALL = f"{IFCB_TRAIN_TEST_DIR_ALL}train/"                  
IFCB_TEST_DIR_ALL = f"{IFCB_TRAIN_TEST_DIR_ALL}test/"                  

IFCB_TRAIN_TEST_DIR_10PCT = f"{TRAIN_TEST_DIR}ifcb/10pct/"                   # path to dir where 10% of the full ifcb tarining dataset (TRAIN_TEST_DIR_ALL) is used. The test dataset remains the same as TRAIN_TEST_DIR_ALL.
IFCB_TRAIN_DIR_10PCT = f"{IFCB_TRAIN_TEST_DIR_10PCT}train/"                  
IFCB_TEST_DIR_10PCT = f"{IFCB_TRAIN_TEST_DIR_10PCT}test/"                  

IFCB_TRAIN_TEST_DIR_BALANCED_1000 = f"{TRAIN_TEST_DIR}ifcb/balanced_1000/"                   
IFCB_TRAIN_DIR_BALANCED_1000 = f"{IFCB_TRAIN_TEST_DIR_BALANCED_1000}train/"                  
IFCB_VALID_DIR_BALANCED_1000 = f"{IFCB_TRAIN_TEST_DIR_BALANCED_1000}valid/"                  
IFCB_TEST_DIR_BALANCED_1000 = f"{IFCB_TRAIN_TEST_DIR_BALANCED_1000}test/"                  
###############################################

################# IFCB SOSIK #################
IFCB_SOSIK_RAW_DIR = f"{DATA_DIR}source/IFCB_Sosik/merged/"  
IFCB_SOSIK_BALANCED_1000_DIR = f"{DATA_DIR}source/IFCB_Sosik/balanced_1000/"  

IFCB_SOSIK_TRAIN_TEST_DIR_ALL = f"{TRAIN_TEST_DIR}ifcb_sosik/all/"                       # path to dir where 100% of the original ifcb_sosik dataset (raw) is split into train-test
IFCB_SOSIK_TRAIN_DIR_ALL = f"{IFCB_SOSIK_TRAIN_TEST_DIR_ALL}train/"                  
IFCB_SOSIK_TEST_DIR_ALL = f"{IFCB_SOSIK_TRAIN_TEST_DIR_ALL}test/"                  

IFCB_SOSIK_TRAIN_TEST_DIR_10PCT = f"{TRAIN_TEST_DIR}ifcb_sosik/10pct/"                   # path to dir where 10% of the full ifcb_sosik tarining dataset (TRAIN_TEST_DIR_ALL) is used. The test dataset remains the same as TRAIN_TEST_DIR_ALL.
IFCB_SOSIK_TRAIN_DIR_10PCT = f"{IFCB_SOSIK_TRAIN_TEST_DIR_10PCT}train/"                  
IFCB_SOSIK_TEST_DIR_10PCT = f"{IFCB_SOSIK_TRAIN_TEST_DIR_10PCT}test/"                  

IFCB_SOSIK_TRAIN_TEST_DIR_BALANCED_1000 = f"{TRAIN_TEST_DIR}ifcb_sosik/balanced_1000/"                  
IFCB_SOSIK_TRAIN_DIR_BALANCED_1000 = f"{IFCB_SOSIK_TRAIN_TEST_DIR_BALANCED_1000}train/"                  
IFCB_SOSIK_VALID_DIR_BALANCED_1000 = f"{IFCB_SOSIK_TRAIN_TEST_DIR_BALANCED_1000}valid/"                  
IFCB_SOSIK_TEST_DIR_BALANCED_1000 = f"{IFCB_SOSIK_TRAIN_TEST_DIR_BALANCED_1000}test/"                  
###############################################

################# IFCB SYKE #################
IFCB_SYKE_RAW_DIR = f"{DATA_DIR}source/IFCB_SYKE/merged/"  
IFCB_SYKE_BALANCED_1000_DIR = f"{DATA_DIR}source/IFCB_SYKE/balanced_1000/"          

IFCB_SYKE_TRAIN_TEST_DIR_BALANCED_1000 = f"{TRAIN_TEST_DIR}ifcb_SYKE/balanced_1000/"                  
IFCB_SYKE_TRAIN_DIR_BALANCED_1000 = f"{IFCB_SYKE_TRAIN_TEST_DIR_BALANCED_1000}train/"                  
IFCB_SYKE_VALID_DIR_BALANCED_1000 = f"{IFCB_SYKE_TRAIN_TEST_DIR_BALANCED_1000}valid/"                  
IFCB_SYKE_TEST_DIR_BALANCED_1000 = f"{IFCB_SYKE_TRAIN_TEST_DIR_BALANCED_1000}test/"                  
###############################################

################### MERGED ##################
MERGED_RAW_DIR = f"{DATA_DIR}source/all/merged/"  
MERGED_BALANCED_1000_DIR = f"{DATA_DIR}source/all/balanced_1000/"  

MERGED_RAW_DIR_ONLY_IFCB = f"{DATA_DIR}source/all/merged_only_ifcb/"  
MERGED_BALANCED_1000_ONLY_IFCB_DIR = f"{DATA_DIR}source/all/balanced_1000_only_ifcb/"  

MERGED_TRAIN_TEST_DIR_ALL = f"{TRAIN_TEST_DIR}merged/all/"                       
MERGED_TRAIN_DIR_ALL = f"{MERGED_TRAIN_TEST_DIR_ALL}train/"                  
MERGED_VALID_DIR_ALL = f"{MERGED_TRAIN_TEST_DIR_ALL}valid/"                  
MERGED_TEST_DIR_ALL = f"{MERGED_TRAIN_TEST_DIR_ALL}test/"                  
MERGED_TRAIN_TEST_DIR_10PCT = f"{TRAIN_TEST_DIR}merged/10pct/"                   
MERGED_TRAIN_DIR_10PCT = f"{MERGED_TRAIN_TEST_DIR_10PCT}train/"                  
MERGED_TEST_DIR_10PCT = f"{MERGED_TRAIN_TEST_DIR_10PCT}test/" 

MERGED_TRAIN_TEST_DIR_BALANCED_1000 = f"{TRAIN_TEST_DIR}merged/balanced_1000/" 
MERGED_BALANCED_1000_TRAIN_DIR = f"{MERGED_TRAIN_TEST_DIR_BALANCED_1000}train/"
MERGED_BALANCED_1000_VALID_DIR = f"{MERGED_TRAIN_TEST_DIR_BALANCED_1000}valid/"
MERGED_BALANCED_1000_TEST_DIR = f"{MERGED_TRAIN_TEST_DIR_BALANCED_1000}test/"


MERGED_TRAIN_TEST_DIR_ONLY_IFCB = f"{TRAIN_TEST_DIR}merged/only_ifcb/"                       
MERGED_TRAIN_DIR_ONLY_IFCB = f"{MERGED_TRAIN_TEST_DIR_ONLY_IFCB}train/"                  
MERGED_VALID_DIR_ONLY_IFCB = f"{MERGED_TRAIN_TEST_DIR_ONLY_IFCB}valid/"                  
MERGED_TEST_DIR_ONLY_IFCB = f"{MERGED_TRAIN_TEST_DIR_ONLY_IFCB}test/"  

MERGED_TRAIN_TEST_DIR_BALANCED_1000_ONLY_IFCB = f"{TRAIN_TEST_DIR}merged/balanced_1000_only_ifcb/" 
MERGED_BALANCED_1000_TRAIN_ONLY_IFCB_DIR = f"{MERGED_TRAIN_TEST_DIR_BALANCED_1000_ONLY_IFCB}train/"
MERGED_BALANCED_1000_VALID_ONLY_IFCB_DIR = f"{MERGED_TRAIN_TEST_DIR_BALANCED_1000_ONLY_IFCB}valid/"
MERGED_BALANCED_1000_TEST_ONLY_IFCB_DIR = f"{MERGED_TRAIN_TEST_DIR_BALANCED_1000_ONLY_IFCB}test/"
###############################################

################### MERGED PLUS ################## existing datasets: ZooScan, IFCB_MAINE, IFCB_SOSIK .... added (3/2) ISIIS_COWEN, IFCB_SYKE 
MERGED_PLUS_RAW_DIR = f"{DATA_DIR}source/all_plus/merged/"  
MERGED_PLUS_BALANCED_1000_DIR = f"{DATA_DIR}source/all_plus/balanced_1000/"   

MERGED_PLUS_TRAIN_TEST_DIR_ALL = f"{TRAIN_TEST_DIR}merged/all_plus/"                       
MERGED_PLUS_TRAIN_DIR_ALL = f"{MERGED_PLUS_TRAIN_TEST_DIR_ALL}train/"                  
MERGED_PLUS_TEST_DIR_ALL = f"{MERGED_PLUS_TRAIN_TEST_DIR_ALL}test/"                  

MERGED_PLUS_TRAIN_TEST_DIR_BALANCED_1000 = f"{TRAIN_TEST_DIR}merged_plus/balanced_1000/" 
MERGED_PLUS_BALANCED_1000_TRAIN_DIR = f"{MERGED_PLUS_TRAIN_TEST_DIR_BALANCED_1000}train/"
MERGED_PLUS_BALANCED_1000_VALID_DIR = f"{MERGED_PLUS_TRAIN_TEST_DIR_BALANCED_1000}valid/"
MERGED_PLUS_BALANCED_1000_TEST_DIR = f"{MERGED_PLUS_TRAIN_TEST_DIR_BALANCED_1000}test/"
###############################################
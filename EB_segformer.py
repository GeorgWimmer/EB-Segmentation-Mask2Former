
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from tqdm import tqdm
import timm
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation, get_scheduler, get_cosine_schedule_with_warmup  ##AdamW,
from EB_transformers_util import WoundDataset, dice_score, AugmentationTransform, compute_class_weights, get_class_value_map, write_masks_from_val_dataset, DiceLoss, ComboLoss, ComboLoss_ignore_background, get_kfolds, CombinedDiceFocalLoss, visualize_class_color_map
from EB_compare_annotations_class_wise import get_prediction_all_metrics_whole_dataset
###############
plt.style.use('bmh')

#####
##Final revised data
#########

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/images/'
# Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/masks/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs4__focaldice_im2mask_aug_jan22/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=4
# Param['LOSS']='Focal+Dice'
# Param['Imagename_is_Maskname']=True
# Param['MORE_AUGMENTATION'] = True
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
# torch.cuda.empty_cache()

# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9961, IoU = 0.9924, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8415, IoU = 0.7737, classname: 1_unaffected
# Class 2: Dice = 0.5616, IoU = 0.4510, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0523, IoU = 0.0312, classname: 5_Blase
# Class 4: Dice = 0.2127, IoU = 0.1498, classname: 6_Kruste
# Class 5: Dice = 0.3211, IoU = 0.2376, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.5848, IoU = 0.4834, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.5100
# ✅ Mean IoU (over present classes): 0.4456
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9635

# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Mean Dice = 0.9961, STD_Dice = 0.0068, ConfidenceIntervall_Dice (95%): [0.9956, 0.9967], classname: Background
# Class 0: Mean Sensitivity = 0.9957, Mean Precision= 0.9967, mean FNR = 0.0043, classname: Background
# Class 1: Mean Dice = 0.8415, STD_Dice = 0.2251, ConfidenceIntervall_Dice (95%): [0.8224, 0.8606], classname: Unaffected
# Class 1: Mean Sensitivity = 0.8976, Mean Precision= 0.8271, mean FNR = 0.1024, classname: Unaffected
# Class 2: Mean Dice = 0.5616, STD_Dice = 0.3073, ConfidenceIntervall_Dice (95%): [0.5354, 0.5878], classname: Epithelialized
# Class 2: Mean Sensitivity = 0.5495, Mean Precision= 0.7392, mean FNR = 0.4505, classname: Epithelialized
# Class 3: Mean Dice = 0.0523, STD_Dice = 0.1163, ConfidenceIntervall_Dice (95%): [0.0302, 0.0744], classname: Blister
# Class 3: Mean Sensitivity = 0.0449, Mean Precision= 0.4268, mean FNR = 0.9551, classname: Blister
# Class 4: Mean Dice = 0.2127, STD_Dice = 0.2703, ConfidenceIntervall_Dice (95%): [0.1849, 0.2405], classname: Crust
# Class 4: Mean Sensitivity = 0.2736, Mean Precision= 0.4144, mean FNR = 0.7264, classname: Crust
# Class 5: Mean Dice = 0.3211, STD_Dice = 0.3116, ConfidenceIntervall_Dice (95%): [0.2894, 0.3528], classname: Erosions
# Class 5: Mean Sensitivity = 0.5453, Mean Precision= 0.4040, mean FNR = 0.4547, classname: Erosions
# Class 6: Mean Dice = 0.5848, STD_Dice = 0.3358, ConfidenceIntervall_Dice (95%): [0.4596, 0.7100], classname: Tumor
# Class 6: Mean Sensitivity = 0.7099, Mean Precision= 0.6836, mean FNR = 0.2901, classname: Tumor
# ✅ Mean Dice (over all present classes): 0.5100, Mean Dice over wound classes (excluding class 0 and 1): 0.3465
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9635
# ✅  Mean Performance measures  over all wound classes (excluding class 0 and 1): Sensitivity:  0.4246, Precision:  0.5336, FNR: 0.5754
#


##eval only using 1024x1024 patches (same as in training)
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/images/'
# Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/masks/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=4
# Param['LOSS']='Focal+Dice'
# Param['Imagename_is_Maskname']=True
# Param['MORE_AUGMENTATION'] = True
#
# Param['eval_only'] = True
# Param['NET_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs4__focaldice_im2mask_aug_jan22/'
# Param['INPUT_SIZE_EVAL']= (1024, 1024)
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs4__focaldice_im2mask_aug_jan22_1024/'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
# torch.cuda.empty_cache()

# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9942, IoU = 0.9886, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8272, IoU = 0.7530, classname: 1_unaffected
# Class 2: Dice = 0.5677, IoU = 0.4475, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0416, IoU = 0.0255, classname: 5_Blase
# Class 4: Dice = 0.2206, IoU = 0.1545, classname: 6_Kruste
# Class 5: Dice = 0.2957, IoU = 0.2126, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.2502, IoU = 0.2043, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.4568
# ✅ Mean IoU (over present classes): 0.3980
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9589





#####
#old unrevised data
#######

#####
##August, 27

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:1"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs1__focaldice_august27/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['LOSS']='Focal+Dice'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# mean overall Dice: 0.48437883341879623
# mean overall IOU: 0.42509386718764
# mean Dice per class : [0.99531089 0.80412088 0.57731298 0.10709718 0.20587693 0.27473987
#  0.55580858]
# mean IOU per class : [0.99084006 0.74166993 0.46216959 0.07653851 0.1446338  0.19963898
#  0.50306902]
#
#
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9948, IoU = 0.9898, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8193, IoU = 0.7402, classname: 1_unaffected
# Class 2: Dice = 0.5591, IoU = 0.4360, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0772, IoU = 0.0543, classname: 5_Blase
# Class 4: Dice = 0.2302, IoU = 0.1632, classname: 6_Kruste
# Class 5: Dice = 0.2897, IoU = 0.2068, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.4552, IoU = 0.3720, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.4893
# ✅ Mean IoU (over present classes): 0.4232
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9555


# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:1"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/CNN_data/data1to1/images/'
# Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/data1to1/masks/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs1__focaldice_im2mask_august27/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['LOSS']='Focal+Dice'
# Param['Imagename_is_Maskname']=True
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)



# mean overall Dice: 0.4532215773458892
# mean overall IOU: 0.4002297696989259
# mean Dice per class : [0.9959435  0.80248746 0.56107909 0.04845257 0.19588849 0.26925336
#  0.43817916]
# mean IOU per class : [0.99202413 0.73911362 0.45177677 0.03101583 0.13825238 0.19755769
#  0.40201582]


# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9954, IoU = 0.9910, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8199, IoU = 0.7433, classname: 1_unaffected
# Class 2: Dice = 0.5332, IoU = 0.4220, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0572, IoU = 0.0374, classname: 5_Blase
# Class 4: Dice = 0.2124, IoU = 0.1511, classname: 6_Kruste
# Class 5: Dice = 0.2670, IoU = 0.1907, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.3934, IoU = 0.3209, classname: 10_Tumor

# ✅ Mean Dice (over present classes): 0.4684
# ✅ Mean IoU (over present classes): 0.4081
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9595

##mit eval only und größerere patchsize
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:1"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/CNN_data/data1to1/images/'
# Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/data1to1/masks/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs1__focaldice_im2mask_august27_eval2048/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['LOSS']='Focal+Dice'
# Param['Imagename_is_Maskname']=True
# Param['eval_only'] = True
# Param['INPUT_SIZE']= (2048, 2048)  #patch size for training and evaluation
# Param['STRIDE'] = (1024, 1024)
# Param['NET_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs1__focaldice_im2mask_august27/'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
#








##mit leichten änderungen von nov12 ander patchsize bei evaluation und gleiche lr bei encoder und decoder
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:1"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs1_nov12_focaldice_august27/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['LOSS']='Focal+Dice'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# mean overall Dice: 0.4820678218818628
# mean overall IOU: 0.42535910090922957
# mean Dice per class : [0.99575504 0.81811034 0.56881261 0.10060926 0.1976074  0.25891815
#  0.56420387]
# mean IOU per class : [0.99175849 0.75835994 0.45560866 0.07215442 0.13835295 0.18369954
#  0.52002387]
#
#
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9953, IoU = 0.9909, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8347, IoU = 0.7603, classname: 1_unaffected
# Class 2: Dice = 0.5533, IoU = 0.4335, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0759, IoU = 0.0533, classname: 5_Blase
# Class 4: Dice = 0.2218, IoU = 0.1566, classname: 6_Kruste
# Class 5: Dice = 0.3001, IoU = 0.2136, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.4246, IoU = 0.3415, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.4865
# ✅ Mean IoU (over present classes): 0.4214
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9574



# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:1"
# Param['IMAGE_DIR']=  '/scratch3/gwimmer/Bilder/EB/CNN_data/data1to1/images/'
# Param['MASK_DIR'] =  '/scratch3/gwimmer/Bilder/EB/CNN_data/data1to1/masks/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_focaldice_im2mask_august27/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] = "nvidia/segformer-b3-finetuned-ade-512-512"
# Param['BATCH_SIZE'] = 1
# Param['LOSS']='Focal+Dice'
# Param['Imagename_is_Maskname']=True
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

####noch mal mit eval_only rechnen, sonst fehler!!!!

# mean overall Dice: 0.4440747367108193
# mean overall IOU: 0.39030918612597265
# mean Dice per class : [0.9946653  0.77143375 0.58803047 0.06111425 0.21025918 0.31223066
#  0.17078956]
# mean IOU per class : [0.98963268 0.70295848 0.46832952 0.0400573  0.15424775 0.23129841
#  0.14564015]
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9940, IoU = 0.9883, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.7735, IoU = 0.6918, classname: 1_unaffected
# Class 2: Dice = 0.5638, IoU = 0.4411, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0466, IoU = 0.0313, classname: 5_Blase
# Class 4: Dice = 0.2494, IoU = 0.1844, classname: 6_Kruste
# Class 5: Dice = 0.3206, IoU = 0.2328, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.2775, IoU = 0.2337, classname: 10_Tumor
# ✅ Mean Dice (over present classes): 0.4608
# ✅ Mean IoU (over present classes): 0.4005
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9537

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:1"
# Param['IMAGE_DIR']=  '/scratch3/gwimmer/Bilder/EB/CNN_data/data1to1/images/'
# Param['MASK_DIR'] =  '/scratch3/gwimmer/Bilder/EB/CNN_data/data1to1/masks/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_focaldice_im2mask_august27_eval/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] = "nvidia/segformer-b3-finetuned-ade-512-512"
# Param['BATCH_SIZE'] = 1
# Param['LOSS']='Focal+Dice'
# Param['Imagename_is_Maskname']=True
# Param['eval_only'] = True
# Param['NET_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_focaldice_im2mask_august27/'
# Param['INPUT_SIZE_EVAL'] = (1800, 1800) #2048 ist memory out bei ameisenbär
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# mean overall Dice: 0.46332809448895307
# mean overall IOU: 0.4078170154752684
# mean Dice per class : [0.9944668  0.77347789 0.58724892 0.04207007 0.2061862  0.29580847
#  0.47510534]
# mean IOU per class : [0.98921975 0.70675256 0.47037967 0.0272424  0.14683956 0.21952921
#  0.43861188]
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9941, IoU = 0.9885, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.7771, IoU = 0.6982, classname: 1_unaffected
# Class 2: Dice = 0.5647, IoU = 0.4456, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0454, IoU = 0.0301, classname: 5_Blase
# Class 4: Dice = 0.2430, IoU = 0.1751, classname: 6_Kruste
# Class 5: Dice = 0.3140, IoU = 0.2276, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.4797, IoU = 0.4077, classname: 10_Tumor
# ✅ Mean Dice (over present classes): 0.4883
# ✅ Mean IoU (over present classes): 0.4247
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9551

#
#
# #####batchsize=4 nach nov13
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/CNN_data/data1to1/images/'
# Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/data1to1/masks/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs4__focaldice_im2mask_august27/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=4
# Param['LOSS']='Focal+Dice'
# Param['Imagename_is_Maskname']=True
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# mean overall Dice: 0.4833547736412874
# mean overall IOU: 0.43816195538236585
# mean Dice per class : [0.99859682 0.72105663 0.73188116 0.         0.13183915 0.37329392
#  0.42681572]
# mean IOU per class : [0.99719945 0.68690899 0.63398481 0.         0.08681149 0.27538254
#  0.38684641]
#
#
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9942, IoU = 0.9888, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8362, IoU = 0.7630, classname: 1_unaffected
# Class 2: Dice = 0.5521, IoU = 0.4396, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0314, IoU = 0.0203, classname: 5_Blase
# Class 4: Dice = 0.2141, IoU = 0.1527, classname: 6_Kruste
# Class 5: Dice = 0.3174, IoU = 0.2333, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.5098, IoU = 0.4333, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.4936
# ✅ Mean IoU (over present classes): 0.4330
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9606


####################August11



####labels_useful_vs_dump big model
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_focaldice_august11/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=2
# Param['LOSS']='Focal+Dice'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# --- Validation Summary over all folds ---
# mean overall Dice: 0.47049010715060036
# mean overall IOU: 0.4135744469508567
# mean Dice per class : [0.99534164 0.79944568 0.58313585 0.08780281 0.18947388 0.29025302
#  0.48158914]
# mean IOU per class : [0.99086855 0.73625817 0.46169169 0.06348636 0.13183    0.20923989
#  0.44754714]



##eval auf ameisenbaer

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:2"
# Param['IMAGE_DIR']=  '/scratch3/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch3/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_focaldice_august11/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=2
# Param['LOSS']='Focal+Dice'
# Param['eval_only']=True
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# ✅ Mean Dice (over present classes): 0.4893
# ✅ Mean IoU (over present classes): 0.4422
# mean overall Dice: 0.47031181463223415
# mean overall IOU: 0.4134467410430052
# mean Dice per class : [0.99534163 0.79944813 0.58312653 0.08780312 0.18946001 0.28901515
#  0.48159954]
# mean IOU per class : [0.99086854 0.73626076 0.46168321 0.06348664 0.13181931 0.20834807
#  0.44756124]
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9947, IoU = 0.9896, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8083, IoU = 0.7271, classname: 1_unaffected
# Class 2: Dice = 0.5737, IoU = 0.4456, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0354, IoU = 0.0253, classname: 5_Blase
# Class 4: Dice = 0.1856, IoU = 0.1287, classname: 6_Kruste
# Class 5: Dice = 0.2714, IoU = 0.1929, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.1940, IoU = 0.1597, classname: 10_Tumor
# ✅ Mean Dice (over present classes): 0.4376
# ✅ Mean IoU (over present classes): 0.3813
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9550






####labels_useful_vs_dump big model mit batchsize=1
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs1__focaldice_august11/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=1
# Param['LOSS']='Focal+Dice'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_bs1__focaldice_august11/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=1
# Param['LOSS']='Focal+Dice'
# Param['Start_FoldNr']=2
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)


# ✅ Mean Dice (over present classes): 0.4727
# ✅ Mean IoU (over present classes): 0.4225
# mean overall Dice: 0.46362489671398044
# mean overall IOU: 0.411066598288708
# mean Dice per class : [0.99638541 0.80881071 0.58377689 0.1572407  0.15621305 0.28890434
#  0.25404318]
# mean IOU per class : [0.99294676 0.77891421 0.47548562 0.11550771 0.10266684 0.20491863
#  0.20702641]

# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9950, IoU = 0.9902, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8223, IoU = 0.7445, classname: 1_unaffected
# Class 2: Dice = 0.5694, IoU = 0.4427, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0708, IoU = 0.0508, classname: 5_Blase
# Class 4: Dice = 0.2231, IoU = 0.1564, classname: 6_Kruste
# Class 5: Dice = 0.2838, IoU = 0.2022, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.4632, IoU = 0.3833, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.4897
# ✅ Mean IoU (over present classes): 0.4243
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9573







####try fehlerhaft wahrscheinlich
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:1"
# Param['IMAGE_DIR']=  '/scratch3/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch3/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump_try_august11/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=1
# Param['LOSS']='Focal+Dice'
# Param['MODEL_NAME'] = "nvidia/segformer-b1-finetuned-ade-512-512"
# Param['RESIZE'] = True  # if images are resized (make them smaller by factor Param['RESIZE_FACTOR'])
# Param['RESIZE_FACTOR'] = 2  # factor by w
# Param['INPUT_SIZE']= (2048, 2048)
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)


# mean overall Dice: 0.4646755859467449
# mean overall IOU: 0.40533719938959084
# mean Dice per class : [0.99547645 0.79161531 0.5788648  0.13958028 0.1815989  0.24037299
#  0.45363861]
# mean IOU per class : [0.99105822 0.72668433 0.45815797 0.09660181 0.12377082 0.17024764
#  0.41375633]
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9948, IoU = 0.9897, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8013, IoU = 0.7196, classname: 1_unaffected
# Class 2: Dice = 0.5671, IoU = 0.4431, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.1159, IoU = 0.0793, classname: 5_Blase
# Class 4: Dice = 0.2226, IoU = 0.1525, classname: 6_Kruste
# Class 5: Dice = 0.2796, IoU = 0.2008, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.3428, IoU = 0.2656, classname: 10_Tumor
# ✅ Mean Dice (over present classes): 0.4749
# ✅ Mean IoU (over present classes): 0.4072
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9542





####labels_useful_vs_dump big model mit resizing von 1280 auf 640
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_focaldice_resize2_1280_august11/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 2
# Param['LOSS']='Focal+Dice'
# Param['RESIZE'] = True  # if images are resized (make them smaller by factor Param['RESIZE_FACTOR'])
# Param['RESIZE_FACTOR'] = 2  # factor by w
# Param['INPUT_SIZE']= (640, 640)
# Param['STRIDE'] = (480, 480)
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# -- Validation Summary over all folds ---
# mean overall Dice: 0.4725872989242904
# mean overall IOU: 0.4168420555880956
# mean Dice per class : [0.99730806 0.81131454 0.57473645 0.20366766 0.18005261 0.27776769
#  0.26326408]
# mean IOU per class : [0.99464692 0.77809783 0.47089979 0.14865923 0.12417778 0.20011807
#  0.20129477]
#
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9956, IoU = 0.9913, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8406, IoU = 0.7657, classname: 1_unaffected
# Class 2: Dice = 0.5816, IoU = 0.4567, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0871, IoU = 0.0597, classname: 5_Blase
# Class 4: Dice = 0.2166, IoU = 0.1531, classname: 6_Kruste
# Class 5: Dice = 0.2955, IoU = 0.2135, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.3356, IoU = 0.2545, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.4789
# ✅ Mean IoU (over present classes): 0.4135
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9594


#
# ###labels_useful_vs_dump big model mit resizing von 2048 auf 1024
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_focaldice_resize2_2048_august11/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['LOSS']='Focal+Dice'
# Param['RESIZE'] = True  # if images are resized (make them smaller by factor Param['RESIZE_FACTOR'])
# Param['RESIZE_FACTOR'] = 2  # factor by w
# Param['INPUT_SIZE']= (1024, 1024)
# Param['accumulation_steps']=2
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)


# ✅ Mean Dice (over present classes): 0.4175
# ✅ Mean IoU (over present classes): 0.3705
# mean overall Dice: 0.4543140071745049
# mean overall IOU: 0.39790031500100975
# mean Dice per class : [0.99500619 0.79726309 0.56471927 0.07210295 0.19214904 0.24156121
#  0.44796074]
# mean IOU per class : [0.99020106 0.73392825 0.44480014 0.05145908 0.13485237 0.1735616
#  0.40048235]
#
#
# --- Validation Metrics over the whole dataset (not seperated in folds) ---
# Class 0: Dice = 0.9943, IoU = 0.9887, classname: dump_classes(background,qual,creme)
# Class 1: Dice = 0.8077, IoU = 0.7265, classname: 1_unaffected
# Class 2: Dice = 0.5645, IoU = 0.4396, classname: 4_(re-) epithelialisiert
# Class 3: Dice = 0.0813, IoU = 0.0579, classname: 5_Blase
# Class 4: Dice = 0.2251, IoU = 0.1593, classname: 6_Kruste
# Class 5: Dice = 0.2847, IoU = 0.2054, classname: Erosionen(7,8,9)
# Class 6: Dice = 0.3402, IoU = 0.2743, classname: 10_Tumor
#
# ✅ Mean Dice (over present classes): 0.4711
# ✅ Mean IoU (over present classes): 0.4074
# ✅ Mean ACC (percentage of correctly classified pixels): 0.9552







######data july 28
################################

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel_july28/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=2
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# Mean Dice (over present classes): 0.4513
# ✅ Mean IoU (over present classes): 0.3983
# mean overall Dice: 0.44817691148280475
# mean overall IOU: 0.38932711282510063
# mean Dice per class : [0.99460374 0.81532201 0.51959035 0.05296547 0.23968553 0.21948035
#  0.29559093]
# mean IOU per class : [0.98939432 0.73661466 0.39971981 0.03622028 0.16830522 0.15191327
#  0.24312224]


# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['IMAGE_DIR']= '/scratch3/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] =  '/scratch3/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_FocalDice_28july/'
# Param['EPOCHS']=500
# Param['Device'] = "cuda:1"
# Param['LOSS']='Focal+Dice'
# Param['MODEL_NAME'] = "nvidia/segformer-b1-finetuned-ade-512-512"
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# mean overall Dice: 0.4643391323845279
# mean overall IOU: 0.40431211506251064
# mean Dice per class : [0.99588423 0.83895976 0.53733578 0.29423124 0.20833365 0.21551244
#  0.27519857]
# mean IOU per class : [0.99189679 0.76663    0.40997966 0.28155466 0.14842289 0.15563432
#  0.20976901]


#############
####data june17

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['IMAGE_DIR']= '/scratch3/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] =  '/scratch3/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_reduced/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_kfold/'
# Param['EPOCHS']=400
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# mean overall Dice: 0.554395346554494
# mean overall IOU: 0.47493250297408346
# mean Dice per class : [0.99519866 0.81072346 0.57616103 0.32301041 0.30264904 0.31862947]
# mean IOU per class : [0.99052585 0.72954011 0.445725   0.22227248 0.21741447 0.24411712]

# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['IMAGE_DIR']= '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_reduced/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_kfold_try/'
# Param['EPOCHS']=500
# Param['Device'] = "cuda:0"
# Param['LOSS']='Focal+Dice'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
#über die ersten 2 folds waren 52% mean dice

######
#
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['IMAGE_DIR']= '/scratch3/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] =  '/scratch3/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_reduced/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_kfold_trysomething/'
# Param['EPOCHS']=400
# Param['INPUT_SIZE']= (1024, 1024)  #patch size for training and evaluation
# Param['STRIDE'] = (768, 768)
# Param['MODEL_NAME'] = "nvidia/segformer-b1-finetuned-ade-512-512"
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
# ✅ Mean Dice (over present classes): 0.4726
# ✅ Mean IoU (over present classes): 0.4194
# mean overall Dice: 0.48644724944387474
# mean overall IOU: 0.42483157421837825
# mean Dice per class : [0.99555733 0.81658971 0.54854717 0.20779204 0.20906708 0.14113017]
# mean IOU per class : [0.9913403  0.7381173  0.42444119 0.14000194 0.14827117 0.10681754]



#
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['IMAGE_DIR']= '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_reduced/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_kfold_big_model/'
# Param['EPOCHS']=500
# Param['Device'] = "cuda:0"
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=2
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
#  Mean Dice (over present classes): 0.4864
# ✅ Mean IoU (over present classes): 0.4332
# mean overall Dice: 0.4977001082310396
# mean overall IOU: 0.43498502218373963
# mean Dice per class : [0.99571078 0.8004827  0.55773968 0.23677836 0.19815162 0.19733751]
# mean IOU per class : [0.991526   0.72200316 0.43578155 0.16447175 0.13649882 0.15962886]

# mean overall Dice: 0.4977001082310396
# mean overall IOU: 0.43498502218373963
# mean Dice per class : [0.99571078 0.8004827  0.55773968 0.23677836 0.19815162 0.19733751]
# mean IOU per class : [0.991526   0.72200316 0.43578155 0.16447175 0.13649882 0.15962886]


#labels_useful
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=['0_background', '1_unaffected', '2_Restklassen(qual,creme)', '4_(re-) epithelialisiert', '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
#mean Dice score sollte etwas unter 40% sein

####labels_useful_vs_dump
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:3"
# Param['IMAGE_DIR']=  '/scratch3/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch3/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__loss_ignore_0/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['LOSS'] = 'weighted_CE+DICE_ignore_background'
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)
##hat nicht funktioniert





####labels_useful_vs_dump big model
# import sys
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
# sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')
# import EB_segformer
# Param={}
# Param['Device'] = "cuda:0"
# Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
# Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_useful_vs_dump/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/segformer_labels_useful_vs_dump__bigmodel/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES']=LABEL_NAMES=['dump_classes(background,qual,creme)', '1_unaffected',  '4_(re-) epithelialisiert',
# '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
# Param['MODEL_NAME'] ='nvidia/segformer-b5-finetuned-ade-640-640'
# Param['BATCH_SIZE'] = 1
# Param['accumulation_steps']=2
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_segformer.EB_seg(Param)

# ✅ Mean Dice (over present classes): 0.4628
# ✅ Mean IoU (over present classes): 0.4040
# mean overall Dice: 0.4726766031896714
# mean overall IOU: 0.40818550087349253
# mean Dice per class : [0.99499551 0.78229055 0.55936531 0.08668662 0.22546489 0.21947072
#  0.57390734]
# mean IOU per class : [0.99014431 0.70030472 0.43243449 0.05996516 0.15521976 0.15392074
#  0.5135767 ]


def EB_seg(params):


    Param = {}
    #data
    Param['Device'] = "cuda:0"
    Param['MASK_DIR'] =  '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_reduced/'  # Path to masks
    Param['IMAGE_DIR']=  '/scratch1/gwimmer/Bilder/EB/images_anonymized/' # Path to images
    Param['OUTPUT_DIR']='' #outputdirectory for segmented images, results and the trained nets
    Param['INPUT_SIZE']= (1024, 1024)  #patch size for training
    Param['STRIDE'] = (1024, 1024) #Stride for evaluation using whole images .Bevor nov12: (896, 896)
    Param['INPUT_SIZE_EVAL']= (2048, 2048) #  nov12 patch size for  evaluation :   mit (2048, 2048) bessere Ergebnisse
    Param['EPOCHS']=500
    #Param['LR']= 3e-5 ##old
    Param['LR'] = 1e-4  ##new with cosine_scheduler
    Param['MODEL_NAME'] = "nvidia/segformer-b0-finetuned-ade-512-512" # nvidia/segformer-b1-finetuned-ade-512-512  , nvidia/segformer-b2-finetuned-ade-512-512,
    # nvidia/segformer-b3-finetuned-ade-512-512 , nvidia/segformer-b4-finetuned-ade-512-512 , nvidia/segformer-b5-finetuned-ade-640-640
    Param['KFOLD']=True
    Param['NR_FOLDS']=4
    Param['BATCH_SIZE'] = 2
    #Param['LOSS']='weighted_CE+DICE' # which loss function, either  'weighted_CE+DICE' or 'Focal+Dice' or  'weighted_CE+DICE_ignore_background'
    Param['LOSS']='Focal+Dice' # which loss function, either  'weighted_CE+DICE' or 'Focal+Dice' or  'weighted_CE+DICE_ignore_background'
    Param['LABEL_NAMES'] = ['dump_classes(background,qual,creme)', '1_unaffected', '4_(re-) epithelialisiert',
                            '5_Blase', '6_Kruste', 'Erosionen(7,8,9)','10_Tumor']
    Param['accumulation_steps']=1 #to artificially increase the batchsize with that factor
    Param['eval_only'] = False  # only evaluation , no training . Must provide trained nets. The trained nets have to be in Param['NET_DIR']
    Param['NET_DIR']=[]
    Param['Start_FoldNr']=0 # if you resume training after a crash, etc.. , you can start again at a specific foldnr
    Param['RESIZE'] = False #if images are resized (make them smaller by factor Param['RESIZE_FACTOR'])
    Param['RESIZE_FACTOR'] =2  # factor by which the images are made smaller. Only plays a role if  Param['RESIZE'] = True
    Param['Imagename_is_Maskname']=False  #if imagenames are equally to masknames, except of ending
    Param['MORE_AUGMENTATION'] = False #additionally scaling, contrast and brightness changes nov20

    #Param['WEIGHT_DECAY']=0.01  ##mit nov 12 wieder entfernt
    #Param['BACKBONE_MULT']=0.1

    for k, v in params.items():
        if k in Param.keys():
            Param[k] = v



    #print(Param)
    for key, val in Param.items():  # Use the Varible names directly instead of Param['VariableName']
        print(key + f'={val}')

    # for key,val in Param.items():   #Use the Varible names directly instead of Param['VariableName']
    #     exec(key + '=val')
    #     print(key + f'={val}')

# # ----------------------------
# # Config
# # ----------------------------
# IMAGE_DIR = '/scratch3/gwimmer/Bilder/EB/images_anonymized/'
# MASK_DIR = '/scratch3/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_reduced/'
# INPUT_SIZE = (1024, 1024)
# BATCH_SIZE = 2
# EPOCHS = 500
# OUTPUT_DIR='/scratch3/gwimmer/Bilder/EB/CNN_segmentations/segformer_kfold/'
#
# #LR = 1e-4
# LR = 3e-5
# DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
# STRIDE=(896, 896)
# KFOLD=True
# NR_FOLDS=4



    DEVICE = torch.device(Param['Device'] if torch.cuda.is_available() else "cpu")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]



    os.makedirs(Param['OUTPUT_DIR'], exist_ok=True)


    transform = AugmentationTransform(crop_size=Param['INPUT_SIZE'], resize=Param['RESIZE'] , resizefactor=Param['RESIZE_FACTOR'])
    if Param['MORE_AUGMENTATION']:
        transform = AugmentationTransform(crop_size=Param['INPUT_SIZE'], resize=Param['RESIZE'] , resizefactor=Param['RESIZE_FACTOR'],
                                          brightness=True, scaling=True, contrast=True)
    else:
        transform = AugmentationTransform(crop_size=Param['INPUT_SIZE'], resize=Param['RESIZE'] , resizefactor=Param['RESIZE_FACTOR'])

    CLASS_VALUE_MAP=get_class_value_map(Param['MASK_DIR'])
    CLASS_COLOR_MAP = {CLASS_VALUE_MAP[k]: (k, k, k) for k in CLASS_VALUE_MAP.keys()}
    NUM_CLASSES=len(CLASS_VALUE_MAP)

    if len(Param['LABEL_NAMES'])>0:
        assert(len(Param['LABEL_NAMES'])==NUM_CLASSES)
        CLASS_NAMES={k: Param['LABEL_NAMES'][ind] for ind,k in enumerate(CLASS_COLOR_MAP.keys())}
        visualize_class_color_map(CLASS_COLOR_MAP, CLASS_NAMES, save_path=os.path.join(Param['OUTPUT_DIR'], 'A_class_colormap.png'))

    dataset = WoundDataset(Param['IMAGE_DIR'], Param['MASK_DIR'], class_map=CLASS_VALUE_MAP, transform=transform, Imagename_is_Maskname=Param['Imagename_is_Maskname'])

    # # weighted classes
    #  data_loader=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # class_weights = compute_class_weights(data_loader, NUM_CLASSES).to(DEVICE)
    # print(class_weights)


    if Param['KFOLD']:
        test_indices_per_fold, train_indices_per_fold=get_kfolds(Param['NR_FOLDS'], dataset)
    else:
        Param['NR_FOLDS']=1
        val_split = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])





    overall_dice=[]
    overall_iou=[]
    Dice_per_class=[]
    IOU_per_class=[]

    for foldnr in range(Param['Start_FoldNr'], Param['NR_FOLDS']):
        print(f"\n  START FOLD NR {foldnr}")

        if Param['KFOLD']:
            val_dataset = torch.utils.data.Subset(dataset, test_indices_per_fold[foldnr])
            train_dataset = torch.utils.data.Subset(dataset, train_indices_per_fold[foldnr])
        else:
            val_split = int(0.8 * len(dataset))
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [val_split, len(dataset) - val_split])
        print(f"\n Length train dataset: {len(train_dataset)},  Length validation dataset: {len(val_dataset)}  ")

        train_loader = DataLoader(train_dataset, batch_size=Param['BATCH_SIZE'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


        if Param['LOSS']=='Focal+Dice':
            criterion = CombinedDiceFocalLoss(dice_weight=1.0, focal_weight=1.0)
        elif Param['LOSS']=='weighted_CE+DICE_ignore_background':
            class_weights = compute_class_weights(train_loader, NUM_CLASSES).to(DEVICE)
            print(class_weights)
            criterion = ComboLoss_ignore_background(class_weights, NUM_CLASSES)

        else: # weighted CE loss combined with DICE loss
            # weighted classes
            class_weights = compute_class_weights(train_loader, NUM_CLASSES).to(DEVICE)
            print(class_weights)
            criterion = ComboLoss(class_weights, NUM_CLASSES)

        model = SegformerForSemanticSegmentation.from_pretrained(
            Param['MODEL_NAME'],
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
        model.to(DEVICE)

        ##old
        # optimizer = torch.optim.AdamW(model.parameters(), lr=Param['LR'])
        # lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
        #                              num_training_steps=Param['EPOCHS'] * len(train_loader) // Param['accumulation_steps'])

        ##nov12
        # optimizer = AdamW([
        #     {"params": model.segformer.encoder.parameters(), "lr": Param['LR'] * Param['BACKBONE_MULT']},
        #     {"params": model.decode_head.parameters(), "lr": Param['LR']},
        # ], weight_decay=Param['WEIGHT_DECAY'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=Param['LR'])
        total_steps = Param['EPOCHS'] * len(train_loader) // Param['accumulation_steps']
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.02 * total_steps),
            num_training_steps=total_steps
        )



        if Param['eval_only']:
            model.load_state_dict(torch.load(Param['NET_DIR'] + "segformer_wound_segmentation_fold" +str(foldnr) + ".pth"))

        else:
            # ----------- TRAINING LOOP ----------- #
            lr_rate_list = []
            train_loss_list = []
            for epoch in range(Param['EPOCHS']):
                model.train()
                total_loss = 0
                loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Param['EPOCHS']}]", leave=False)
                step=0
                for images, masks in loop:
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)

                    outputs = model(pixel_values=images)
                    logits = outputs.logits  # shape: (B, C, H, W)
                    logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
                    loss = criterion(logits, masks)

                    loss.backward()
                    # Clip gradients (optional, but recommended)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if (step + 1) % Param['accumulation_steps'] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                    step=step+1


                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

                print(f"Epoch {epoch+1} Training Loss: {total_loss / len(train_loader):.4f}")
                train_loss_list.append(total_loss / len(train_loader))
                lr_rate_list.append(lr_scheduler.get_lr())

                # ------- Evaluation on val set -------- #
                if epoch%10==0 or epoch==Param['EPOCHS']-1:
                    model.eval()
                    with torch.no_grad():
                        iou_sum = 0
                        count = 0
                        for images, masks in val_loader:
                            images = images.to(DEVICE)
                            masks = masks.to(DEVICE)
                            outputs = model(pixel_values=images)
                            logits = outputs.logits  # shape: (B, C, H, W)
                            logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
                            preds = torch.argmax(logits, dim=1)
                            intersection = torch.logical_and(preds == masks, masks > 0)
                            union = torch.logical_or(preds == masks, masks > 0)
                            iou = intersection.sum().float() / (union.sum().float() + 1e-6)
                            iou_sum += iou.item()
                            count += 1
                        print(f"Validation mIoU: {iou_sum / count:.4f}")
            # ---------------------------------------- #
            torch.save(model.state_dict(), os.path.join(Param['OUTPUT_DIR'],"segformer_wound_segmentation_fold" +str(foldnr) + ".pth"))

            # show learning rate
            plt.figure(figsize=(10, 10))
            plt.plot([i[0] for i in lr_rate_list])
            plt.ylabel('learing rate during training in fold ' + str(foldnr), fontsize=22)
            plt.savefig(Param['OUTPUT_DIR'] + '_LR_plot_' + str(foldnr) + '.png')
            plt.show()

            # show losses
            plt.figure(figsize=(10, 10))
            plt.plot(train_loss_list, marker='o', label="Training Loss")
            plt.ylabel('loss in fold ' + str(foldnr), fontsize=22)
            plt.legend()
            plt.savefig(Param['OUTPUT_DIR'] + '_Loss_plot_' + str(foldnr) + '.png')
            plt.show()


        #Write out full images of the evaluation set
        CLASS_COLOR_MAP  = {CLASS_VALUE_MAP[k]: (k, k, k) for k in  CLASS_VALUE_MAP.keys()}
        fold_overall_dice, fold_overall_iou, fold_Dice_per_class, fold_IOU_per_class = write_masks_from_val_dataset(val_dataset=val_dataset,
                                     model=model, output_dir=Param['OUTPUT_DIR'], class_colors=CLASS_COLOR_MAP,
                                     device=DEVICE, num_classes= NUM_CLASSES, crop_size=Param['INPUT_SIZE_EVAL'], stride=Param['STRIDE'], foldnr=foldnr,
                                     label_names=Param['LABEL_NAMES'], resize=Param['RESIZE'] , resizefactor=Param['RESIZE_FACTOR'])


        overall_dice.append(fold_overall_dice)
        overall_iou.append(fold_overall_iou)
        Dice_per_class.append(fold_Dice_per_class)
        IOU_per_class.append(fold_IOU_per_class)


    mean_dice = np.mean(overall_dice)
    mean_iou = np.mean(overall_iou)
    mean_Dice_per_class=np.mean(Dice_per_class,0)
    mean_IOU_per_class=np.mean(IOU_per_class,0)
    results_summary = []
    message=f"mean overall Dice: {mean_dice}"
    print(message)
    results_summary.append(message)
    message=f"mean overall IOU: {mean_iou}"
    print(message)
    results_summary.append(message)
    message=f"mean Dice per class : {mean_Dice_per_class}"
    print(message)
    results_summary.append(message)
    message=f"mean IOU per class : {mean_IOU_per_class}"
    print(message)
    results_summary.append(message)

    # Write only final summary to file
    results_summary_without_folds, cm_fig = get_prediction_all_metrics_whole_dataset(Param['MASK_DIR'], Param['OUTPUT_DIR'],
                                                                         Param['LABEL_NAMES'], Imagename_is_Maskname=Param['Imagename_is_Maskname'])
    cm_fig.savefig(os.path.join(Param['OUTPUT_DIR'], 'ConfusionMatrix.png'), dpi=300)
    log_file = os.path.join(Param['OUTPUT_DIR'], "results.txt")
    with open(log_file, "a") as log:
        log.write("\n--- Validation Summary over all folds ---\n")
        for line in results_summary:
            log.write(line + "\n")
        for line in results_summary_without_folds:
            log.write(line + "\n")


    return mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class


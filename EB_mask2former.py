import sys
sys.path.insert(1, '/home/pma/gwimmer/texture/python/')
sys.path.insert(1, '/home/pma/gwimmer/texture/python/EB/')


import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.optim import AdamW
import glob
plt.style.use('bmh')

from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation,  get_cosine_schedule_with_warmup  ### AdamW,
from EB_transformers_util import get_class_value_map,  get_kfolds, AugmentationTransform_mask2former, collate_fn_mask2former, WoundDataset_mask2former,  write_masks_from_val_dataset_mask2former, visualize_class_color_map, infer_large_image_mask2former
from EB_compare_annotations_class_wise import get_prediction_all_metrics_whole_dataset



#####
##Training  script for the Mask2former on our EB dataset
#########
#For training and evaluation, the image and masknames must be of the form  'Pat' + PatientID + '_' + ImageID + '_' + AnnotatorID +.jpg
#e.g.: 'Pat01_a19A_christian.jpg'

###first go t0 the directory with the code
# import EB_mask2former
# Param={}
# Param['MASK_DIR'] ='/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/masks/'
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/mask2former_labels_useful_vs_dump_ignore_class0_im2mask_bs4_aug_jan22/'
# Param['IMAGE_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/images/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES'] = ['Background,qual,creme', 'Unaffected', '(Re-)Epithelialized', 'Blister', 'Crust', 'Erosions', 'Tumor']
# Param['IGNORE_INDEX']=0  #ignore  class background for loss computation
# Param['DEVICE']='cuda:0'
# Param['RESIZE'] = {'height': 512, 'width': 512}
# Param['BATCH_SIZE'] = 2
# Param['accumulation_steps']=2
# Param['Imagename_is_Maskname']=True
# Param['MORE_AUGMENTATION'] = True
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_mask2former.EB_seg(Param)


#####
##Evaluation  script for the Mask2former on our EB dataset (bigger patchsize as for training)
#########

###first go t0 the directory with the code
# import EB_mask2former
# Param={}
# Param['MASK_DIR'] ='/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/masks/'
# Param['IMAGE_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/Final/data1to1/images/'
# Param['EPOCHS']=500
# Param['LABEL_NAMES'] = ['Background,qual,creme', 'Unaffected', '(Re-)Epithelialized', 'Blister', 'Crust', 'Erosions', 'Tumor']
# Param['IGNORE_INDEX']=0  #ignore  class background for loss computation
# Param['DEVICE']='cuda:1'
# Param['BATCH_SIZE'] = 2
# Param['accumulation_steps']=2
# Param['Imagename_is_Maskname']=True
# Param['OUTPUT_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/mask2former_labels_useful_vs_dump_ignore_class0_im2mask_bs4_aug_jan22_eval/'
# Param['eval_only']=True
# Param['NET_DIR']='/scratch1/gwimmer/Bilder/EB/CNN_segmentations/mask2former_labels_useful_vs_dump_ignore_class0_im2mask_bs4_aug_jan22/'
# Param['RESIZE'] = {'height': 1024, 'width': 1024}
# Param['INPUT_SIZE'] = (2048, 2048)
# mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_mask2former.EB_seg(Param)
# torch.cuda.empty_cache()

#################



def EB_seg(params):



    Param = {}

    Param['IMAGE_DIR'] = '/scratch1/gwimmer/Bilder/EB/images_anonymized/'
    Param['MASK_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_data/medical_annotations_processed/labels_reduced/'
    Param['INPUT_SIZE'] = (1024, 1024)
    Param['BATCH_SIZE'] = 2
    Param['LABEL_NAMES']= ['Background,qual,creme', 'Unaffected', '(Re-)Epithelialized', 'Blister', 'Crust', 'Erosions', 'Tumor']
    Param['OUTPUT_DIR'] = '/scratch1/gwimmer/Bilder/EB/CNN_segmentations/mask2former/'
    Param['NR_FOLDS']=4
    Param['DEVICE']='cuda:1'
    Param['EPOCHS']=500
    Param['LR']=1e-5
    #STRIDE=896
    Param['STRIDE']=768
    Param['MODEL_NAME']="facebook/mask2former-swin-large-ade-semantic"
    Param['RESIZE']={'height': 512, 'width': 512} #nov 20, zuvor:  Param['RESIZE']={'height': 384, 'width': 384}
    Param['IGNORE_INDEX']=0 #which inex should be ignored (the nth index of the labels in Label_NAMES (e.g 0 for background or 5 for Restklassen for labels_reduced)
                                # 255 is just a placeholder that does not occur in the dataset
    Param['eval_only']=False #only evaluation , no training . Must provide trained nets. The trained nets have to be in Param['NET_DIR']
    Param['NET_DIR']=[]
    Param['Start_FoldNr']=0 # if you resume training after a crash, etc.. , you can start again at a specific foldnr
    Param['Imagename_is_Maskname']=False  #if imagenames are equally to masknames, except of ending
    Param['accumulation_steps']=1 ##to artificially increase the batchsize with that factor

    Param['MORE_AUGMENTATION']=True #additionally scaling, contrast and brightness changes
    Param['WEIGHT_DECAY'] = 0.01

    #########

    for k, v in params.items():
        if k in Param.keys():
            Param[k] = v



    #print(Param)
    for key, val in Param.items():  # Use the Varible names directly instead of Param['VariableName']
        print(key + f'={val}')






    os.makedirs(Param['OUTPUT_DIR'], exist_ok=True)

    device = torch.device(Param['DEVICE'] if torch.cuda.is_available() else "cpu")
    CLASS_VALUE_MAP=get_class_value_map(Param['MASK_DIR'])
    CLASS_COLOR_MAP  = {CLASS_VALUE_MAP[k]: (k, k, k) for k in  CLASS_VALUE_MAP.keys()}
    NUM_CLASSES=len(CLASS_VALUE_MAP)
    # if len(Param['LABEL_NAMES'])>0:
    #     id2label = {ind: lab for ind, lab in enumerate(Param['LABEL_NAMES'])}
    #     id2label_mask = {list(CLASS_VALUE_MAP.keys())[ind] for ind, lab in enumerate(Param['LABEL_NAMES'])}
    if len(Param['LABEL_NAMES'])>0:
        assert(len(Param['LABEL_NAMES'])==NUM_CLASSES)
        CLASS_NAMES={k: Param['LABEL_NAMES'][ind] for ind,k in enumerate(CLASS_COLOR_MAP.keys())}
        visualize_class_color_map(CLASS_COLOR_MAP, CLASS_NAMES, save_path=os.path.join(Param['OUTPUT_DIR'], 'A_class_colormap.png'))


    processor = Mask2FormerImageProcessor.from_pretrained(
        Param['MODEL_NAME'], reduce_labels=False
    )
    processor.size= Param['RESIZE'] #other sizes for resizing of the processor

    if Param['MORE_AUGMENTATION']:
        transform = AugmentationTransform_mask2former(crop_size=Param['INPUT_SIZE'],
                                          brightness=True, scaling=True, contrast=True)
    else:
        transform = AugmentationTransform_mask2former(crop_size=Param['INPUT_SIZE'])


    dataset=WoundDataset_mask2former(Param['IMAGE_DIR'], Param['MASK_DIR'], processor, CLASS_VALUE_MAP,  transform=transform, ignore_index=Param['IGNORE_INDEX'], Imagename_is_Maskname=Param['Imagename_is_Maskname'])
    test_indices_per_fold, train_indices_per_fold=get_kfolds(Param['NR_FOLDS'], dataset)

    overall_dice = []
    overall_iou = []
    Dice_per_class = []
    IOU_per_class = []

    for foldnr in range(Param['Start_FoldNr'], Param['NR_FOLDS']):
        print(f"\n  START FOLD NR {foldnr}")

        val_dataset = torch.utils.data.Subset(dataset, test_indices_per_fold[foldnr])
        train_dataset = torch.utils.data.Subset(dataset, train_indices_per_fold[foldnr])
        print(f"\n Length train dataset: {len(train_dataset)},  Length validation dataset: {len(val_dataset)}  ")

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            Param['MODEL_NAME'],
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        ).to(device)







        train_loader = DataLoader(train_dataset, batch_size=Param['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn_mask2former, num_workers=4)
        #val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_mask2former)


        optimizer = AdamW(model.parameters(), lr=Param['LR'], weight_decay=Param['WEIGHT_DECAY'])

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * Param['EPOCHS']),
            num_training_steps=Param['EPOCHS'])

        if Param['eval_only']:
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                Param['MODEL_NAME'],
                num_labels=NUM_CLASSES,
                ignore_mismatched_sizes=True
            ).to(device)
            model.load_state_dict(torch.load(Param['NET_DIR'] + "mask2former_wound_segmentation_fold" +str(foldnr) + ".pth"))

        else:
            # ------------- Training ------------
            lr_rate_list = []
            train_loss_list = []
            for epoch in range(Param['EPOCHS']):
                model.train()
                total_loss = 0
                step=0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                    outputs = model(
                        pixel_values=batch["pixel_values"].to(device),
                        mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
                        class_labels=[labels.to(device) for labels in batch["class_labels"]],
                    )


                    loss = outputs.loss
                    loss.backward()
                    # Clip gradients (optional, but recommended)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if (step + 1) % Param['accumulation_steps'] == 0:
                        optimizer.step()
                        #scheduler.step()
                        optimizer.zero_grad()
                    step=step+1
                    total_loss += loss.item()
                print(f"Epoch {epoch+1}: loss = {total_loss/len(train_loader):.4f}")
                scheduler.step()
                lr_rate_list.append(scheduler.get_lr())
                train_loss_list.append(total_loss / len(train_loader))

            torch.save(model.state_dict(), os.path.join(Param['OUTPUT_DIR'],"mask2former_wound_segmentation_fold" +str(foldnr) + ".pth"))

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



        # ------------- Inference & Save Masks -------------



        fold_overall_dice, fold_overall_iou, fold_Dice_per_class, fold_IOU_per_class=write_masks_from_val_dataset_mask2former(val_dataset, model,
                            Param['OUTPUT_DIR'], processor, CLASS_COLOR_MAP, device=device, num_classes=NUM_CLASSES,
                                         patch_size=Param['INPUT_SIZE'][0], stride=Param['STRIDE'], foldnr=foldnr, label_names=Param['LABEL_NAMES'])





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
    cm_fig.savefig(os.path.join(Param['OUTPUT_DIR'],'ConfusionMatrix.png'), dpi=300)
    log_file = os.path.join(Param['OUTPUT_DIR'], "results.txt")
    with open(log_file, "a") as log:
        log.write("\n--- Validation Summary over all folds ---\n")
        for line in results_summary:
            log.write(line + "\n")
        for line in results_summary_without_folds:
            log.write(line + "\n")

    return mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class




    #################################################

###2026

# ###first got to the directory with the code
# Param = {}
# Param['OUTPUT_DIR'] =  '/scratch1/gwimmer/Bilder/EB/kauba/'
# Param['IMAGE_DIR'] = '/lab/wavelab-datasets/medical/ServEB/3D Modelle/'
# Param['RESIZE'] = {'height': 1024, 'width': 1024}
# Param['INPUT_SIZE'] = (2048, 2048)
# Param['MultipleDirectories']=False ##if the images are in multiple subfolders
# Param['DEVICE'] = 'cuda:0'
# import EB_mask2former
# EB_mask2former.EB_validate(Param)



def EB_validate(params):


    Param = {}
    #Param['NETDIR']=  '/scratch1/gwimmer/Bilder/EB/CNN_segmentations/mask2former_labels_useful_vs_dump_ignore_class0_bs1_august11/mask2former_wound_segmentation_fold0.pth'  #2025
    Param['NETDIR']=   '/scratch1/gwimmer/Bilder/EB/CNN_segmentations/mask2former_labels_useful_vs_dump_ignore_class0_im2mask_bs4_aug_jan22/mask2former_wound_segmentation_fold0.pth'  #2026
    Param['MODEL_NAME'] = "facebook/mask2former-swin-large-ade-semantic"
    Param['IMAGE_DIR'] = '/scratch1/gwimmer/Bilder/EB/Vectra_WB360/4/'
    Param['INPUT_SIZE'] = (2048, 2048)
    Param['LABEL_NAMES'] = ['Background,qual,creme', 'Unaffected', '(Re-)Epithelialized', 'Blister', 'Crust', 'Erosions', 'Tumor']
    # Param['IGNORE_INDEX'] = 0  # ignore  dump klasse
    Param['DEVICE'] = 'cuda:1'
    Param['RESIZE'] = {'height': 1024, 'width': 1024}
    Param['BATCH_SIZE'] = 1
    Param['OUTPUT_DIR'] = '/scratch1/gwimmer/Bilder/EB/Vectra_WB360/mask/4/'
    Param['STRIDE']=512
    Param['MultipleDirectories']=False

    Param['IGNORE_INDEX']=0 #which inex should be ignored (the nth index of the labels in Label_NAMES (e.g 0 for background or 5 for Restklassen for labels_reduced)
                                # 255 is just a placeholder that does not occur in the dataset




    #########

    for k, v in params.items():
        if k in Param.keys():
            Param[k] = v

    device=Param['DEVICE']
    NUM_CLASSES=len(Param['LABEL_NAMES'])
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        Param['MODEL_NAME'],
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    model.load_state_dict(torch.load(Param['NETDIR']))


    processor = Mask2FormerImageProcessor.from_pretrained(
        Param['MODEL_NAME'], reduce_labels=False
    )
    processor.size = Param['RESIZE']  # other sizes for resizing of the processor



    if Param['MultipleDirectories']:
        imgs = list(sorted(glob.glob(Param['IMAGE_DIR']+ '/*/' +'*.jpg' )))
    else:
        # imgs = list(sorted(glob.glob(Param['IMAGE_DIR']+ '*' )))
        imgs = list(sorted(glob.glob(Param['IMAGE_DIR'] + '*.jpg')))  # Changed feb 2026



    for img_path in imgs:
    
        print(f'Process image {img_path}')
        original_image = Image.open(img_path).convert("RGB")

        resized=False
        width, height = original_image.size
        patchsize=Param['INPUT_SIZE'][0]
        if width<patchsize or height<patchsize:  #if the image is smaller that the patchsize
            new_width = max(patchsize, width)
            new_height = max(patchsize, height)

            padded_img = Image.new('RGB', (new_width, new_height), (0,0,0))
            padded_img.paste(original_image, (0, 0))
            original_image = padded_img
            resized=True

            print(f'image {img_path} has been resized for processing')


        pred_mask = infer_large_image_mask2former(
            image=original_image,
            model=model,
            processor=processor,
            device=device,
            patch_size=Param['INPUT_SIZE'][0],
            stride=Param['STRIDE'],
        )
    
        pred_mask = pred_mask *(255/(NUM_CLASSES-1))
        pred_mask =np.uint8(pred_mask)
    

        if Param['MultipleDirectories']:
            dirname=os.path.basename(os.path.dirname(img_path))
            dirout=os.path.join(Param['OUTPUT_DIR'], dirname)
            if not os.path.exists(dirout):
                os.makedirs(dirout)
            path_out=os.path.join(dirout, os.path.basename(img_path)[:-3] +'png')
        else:
            path_out=os.path.join(Param['OUTPUT_DIR'], os.path.basename(img_path)[:-3] +'png')

        print(f'Write processed to {path_out}')

        #Image.fromarray(pred_mask).save(path_out)
        pred_mask = Image.fromarray(pred_mask)
        if resized:
            pred_mask = pred_mask.crop((0, 0, width, height))
        pred_mask.save(path_out)


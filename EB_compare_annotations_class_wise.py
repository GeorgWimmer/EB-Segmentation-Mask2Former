

import os
from PIL import Image
import numpy as np
import glob
import cv2
import shutil
from scipy.stats import mode
#from EB_get_segmentation_mask import gaussian_kernel, visualize_segmentation
from itertools import combinations
from EB_transformers_util import  get_class_value_map
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns



def get_indices(element, lst):
    indices = []
    for i in range(len(lst)):
        if lst[i] == element:
            indices.append(i)
    return indices



def compute_metrics(pred, target, num_classes):
    dice_scores = []
    ious = []

    for cls in range(num_classes):
        target_inds = (target == cls)
        pred_inds = (pred == cls)

        target_area = target_inds.sum()
        pred_area = pred_inds.sum()
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        # ✅ Skip only if class is not present in either of the two comparison images
        if target_area == 0 and pred_area ==0:
        #if target_area == 0 or pred_area ==0:  #wenn man die gleichen resultate haben will wie bei EB_compare_annotation: Vergleiche nur wenn beide Bilder die Klasse enthalten  ---nicht richtig
            continue

        dice = (2. * intersection) / (pred_area + target_area + 1e-8)
        iou = intersection / (union + 1e-8)

        dice_scores.append((cls, dice))
        ious.append((cls, iou))

    return dice_scores, ious



def compute_metrics_all_dice(pred, target, num_classes):
    dice_scores = []
    ious = []

    for cls in range(num_classes):
        target_inds = (target == cls)
        pred_inds = (pred == cls)

        target_area = target_inds.sum()
        pred_area = pred_inds.sum()
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if target_area == 0:
            continue

        dice = (2. * intersection + 1e-8) / (pred_area + target_area + 1e-8)
        iou = (intersection  + 1e-8) / (union + 1e-8)

        dice_scores.append((cls, dice))
        ious.append((cls, iou))

    return dice_scores, ious





def compute_all_metrics(pred, target, num_classes):
    dice_scores = []
    dice_scores_include_empty_gt_pred=[]
    ious = []
    sensitivities = []
    precisions = []
    #fprs = []
    fnrs = []

    test_flattened = pred.flatten()
    reference_flattened = target.flatten()
    cm = confusion_matrix(reference_flattened, test_flattened, labels=list(range(num_classes)))


    for cls in range(num_classes):
        reference = (target == cls)
        prediction = (pred == cls)

        tp = int(((prediction != 0) * (reference != 0)).sum())
        fp = int(((prediction != 0) * (reference == 0)).sum())
        tn = int(((prediction == 0) * (reference == 0)).sum())
        fn = int(((prediction == 0) * (reference != 0)).sum())
        #size = int(np.prod(reference.shape, dtype=np.int64))

        #Dice scores
        if (tp == 0 and fp == 0 and fn == 0):  #both the prediction and the ground truth do not include the class
            ##dice_scores:do nothing for dice score because standard dice score is not defined.
            # dice_scores_include_empty_gt_pred: Add aperfect score (1.0) for dice_scores_include_empty_gt_pred (it was correctly
            # predicted that the class does not appear in that image)
            dice_scores_include_empty_gt_pred.append((cls,float(1.0)))
        else:
            DSC=float(2.0 * tp / (2 * tp + fp + fn))
            dice_scores.append((cls,DSC))
            dice_scores_include_empty_gt_pred.append((cls,DSC))

        #Precision: only when model prediction includes the class
        if (tp+fp) > 0:
            precision=float(tp / (tp + fp))
            precisions.append((cls, precision))

        #Sensitivity (=Recall): only computed when class is present in the groundtruth
        if (tp+fn) > 0:
            sensitivity=float(tp / (tp + fn))
            sensitivities.append((cls, sensitivity))

            #False-Negative Rate:  only computed when class is present in the groundtruth
            fnr=float(fn / (tp + fn))
            fnrs.append((cls, fnr))




    return dice_scores, dice_scores_include_empty_gt_pred, sensitivities, precisions, fnrs, cm


##alte version: verwende: print_annotation_metrics_per_annotator_corrected
def print_annotation_metrics(mask_root, labels):
    masks = glob.glob(mask_root + '*.png')

    masks.sort()
    mask_names = []
    for m in masks:
        mask_names.append(os.path.basename(m))

    mname = []
    annotator = []
    for m in mask_names:
        ind_last = m.rfind('_')
        annotator.append(m[ind_last + 1:-4])
        mname.append(m[:ind_last])

    mnames = list(set(mname))
    mnames.sort()

    annotators = list(set(annotator))
    annotator_scores = annotator_scores = {a: [] for a in annotators}
    class_map = get_class_value_map(mask_root)
    num_classes = len(class_map)

    dice_per_class = defaultdict(list)
    iou_per_class = defaultdict(list)
    ACC=[]
    dice_multi_class= []
    for m in mnames:
        indices = get_indices(m, mname)
        annos = [annotator[i] for i in indices]

        if len(indices) == 2:
            pred1 = cv2.imread(masks[indices[0]], cv2.IMREAD_GRAYSCALE)
            pred2 = cv2.imread(masks[indices[1]], cv2.IMREAD_GRAYSCALE)

            acc = np.sum(pred1 == pred2) / np.sum(pred1 < 10000)
            ACC.append(acc)

            for raw_val, class_idx in class_map.items():
                pred1[pred1 == raw_val] = class_idx
                pred2[pred2 == raw_val] = class_idx

            dice_scores, iou_scores = compute_metrics(pred1, pred2, num_classes)
            for cls, dice in dice_scores:
                dice_per_class[cls].append(dice)
            for cls, iou in iou_scores:
                iou_per_class[cls].append(iou)

            MCdice = 2 * np.sum(np.logical_and(pred1 == pred2, pred1 > 0)) / (np.sum(pred1 > 0) + np.sum(pred2 > 0))
            dice_multi_class.append(MCdice)

        elif len(indices) > 2:
            pairs = list(combinations(indices, 2))
            for p in pairs:
                pred1 = cv2.imread(masks[p[0]], cv2.IMREAD_GRAYSCALE)
                pred2 = cv2.imread(masks[p[1]], cv2.IMREAD_GRAYSCALE)

                acc = np.sum(pred1 == pred2) / np.sum(pred1 < 10000)
                ACC.append(acc)

                for raw_val, class_idx in class_map.items():
                    pred1[pred1 == raw_val] = class_idx
                    pred2[pred2 == raw_val] = class_idx

                dice_scores, iou_scores = compute_metrics(pred1, pred2, num_classes)
                for cls, dice in dice_scores:
                    dice_per_class[cls].append(dice)
                for cls, iou in iou_scores:
                    iou_per_class[cls].append(iou)

                MCdice = 2 * np.sum(np.logical_and(pred1 == pred2, pred1 > 0)) / (np.sum(pred1 > 0) + np.sum(pred2 > 0))
                dice_multi_class.append(MCdice)

    Dice_per_class = []
    IOU_per_class = []
    print("\n--- Validation Metrics ---")
    for cls in range(num_classes):
        if cls in dice_per_class:
            mean_dice = np.mean(dice_per_class[cls])
            std_dice= np.std(dice_per_class[cls])
            mean_iou = np.mean(iou_per_class[cls])
            Dice_per_class.append(mean_dice)
            IOU_per_class.append(mean_iou)
            #print(f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, classname: {labels[cls]}")
            comparisons=len(dice_per_class[cls])
            print(f"Class {cls}: Dice = {mean_dice:.4f} (std ={std_dice:.4f}) , IoU = {mean_iou:.4f}, compared pairs ={comparisons}, classname: {labels[cls]}")
        else:
            print(f"Class {cls}: Dice = N/A, IoU = N/A (not present),  classname: {labels[cls]}")

    overall_dice = np.mean([np.mean(scores) for scores in dice_per_class.values()])
    overall_iou = np.mean([np.mean(scores) for scores in iou_per_class.values()])
    overall_ACC = np.mean(ACC)
    MultiClassDice=np.mean(dice_multi_class)
    MultiClassDice_std=np.std(dice_multi_class)
    print(f"\n✅ Mean Dice (over present classes): {overall_dice:.4f}")
    print(f"\n✅ Multi Class Dice Score: {MultiClassDice:.4f}  (std={MultiClassDice_std:.4f})")
    print(f"✅ Mean IoU (over present classes): {overall_iou:.4f}")
    print(f"✅ Mean ACC (percentage of correctly classified pixels): {overall_ACC:.4f}")



#####Compute all types of prediction metrics (Includung precission, recall confidence intervall,...=


def get_prediction_all_metrics_whole_dataset(mask_gt, mask_pred_root, labels, Imagename_is_Maskname=False):
    # ##Inputs:   path to the grountruth masks: mask_gt (only png)
    #             path to the predicted masks: mask_pred_root
    #             names of the classes_ labels
    #              Imagename_is_Maskname...if images and masks have the same name (if not them images and mask must be of to following form: image: imagename.png mask:imagename_annotator.png)
    # Output:     All results (Dice, ACC, Sensitivity, Precision, False-Negative Rate (FNR), Confusion Matrix) per class and in total in the form of a string


    masks = glob.glob(mask_gt + '*.png')

    class_map = get_class_value_map(mask_gt)
    num_classes = len(class_map)

    ###this following part is only necessary if the masks and images have different names (Imagename_is_Maskname=False)
    masks.sort()
    mask_names = []
    for m in masks:
        mask_names.append(os.path.basename(m))
    mname = []
    annotator = []
    for m in mask_names:
        ind_last = m.rfind('_')
        #annotator.append(m[ind_last + 1:-4])
        mname.append(m[:ind_last])


    dice_per_class = defaultdict(list)
    #iou_per_class = defaultdict(list)
    dice_ie_per_class = defaultdict(list)
    sensitivity_per_class = defaultdict(list)
    precision_per_class = defaultdict(list)
    fnr_per_class = defaultdict(list)
    ACC=[]
    global_confusion_matrix=np.zeros([num_classes,num_classes])


    for ind,m in enumerate(masks):
        gt=cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        #path_pred=glob.glob(os.path.join(mask_pred_root, mname[ind])+'*')
        #path_pred=path_pred[0] ##There may be multiple gt's from the same image, but the AI-based prediction of the same image is always the same
        if Imagename_is_Maskname:
            path_pred=os.path.join(mask_pred_root, os.path.basename(m))
        else:
            path_pred=glob.glob(os.path.join(mask_pred_root, mname[ind])+'*')
            path_pred=path_pred[0] ##There may be multiple gt's from the same image, but the AI-based prediction of the same image is always the same
        pred=cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        acc=np.sum(gt==pred)/np.sum(gt<10000)
        ACC.append(acc)

        for raw_val, class_idx in class_map.items():
            gt[gt == raw_val] = class_idx
            pred[pred == raw_val] = class_idx
        dice_scores, dice_scores_include_empty_gt_pred, sensitivities, precisions, fnrs, cm= compute_all_metrics(pred, gt, num_classes)
        #dice_scores, iou_scores = compute_metrics_corrected(pred, gt, num_classes)  ###CORRECTED only images that include the class in the gt

        for cls, dice in dice_scores:
            dice_per_class[cls].append(dice)
        # for cls, iou in iou_scores:
        #         #     iou_per_class[cls].append(iou)
        for cls, dice in dice_scores_include_empty_gt_pred:
            dice_ie_per_class[cls].append(dice)
        for cls, sens in sensitivities:
            sensitivity_per_class[cls].append(sens)
        for cls, prec in precisions:
            precision_per_class[cls].append(prec)
        for cls, fnr in fnrs:
            fnr_per_class[cls].append(fnr)
        global_confusion_matrix=global_confusion_matrix+cm

    results_summary = []
    ##mean values over the dataset (with capital first letter)
    Dice_per_class = []
    #IOU_per_class = []
    Sensitivity_per_class=[]
    Precision_per_class=[]
    Fnr_per_class=[]
    message = "\n--- Validation Metrics over the whole dataset (not seperated in folds) ---"
    print(message)
    results_summary.append(message)

    # # Normalize rows of the confusion matrix to show percentages (0.0 to 1.0)
    # global_cm_normalized = global_confusion_matrix.astype('float') / global_confusion_matrix.sum(axis=1)[:, np.newaxis]

    row_sums = global_confusion_matrix.sum(axis=1, keepdims=True)
    cm_normalized = np.zeros_like(global_confusion_matrix, dtype=float)
    nonzero_rows = (row_sums > 0).flatten()
    cm_normalized[nonzero_rows] = (global_confusion_matrix[nonzero_rows] / row_sums[nonzero_rows]) * 100
    entries = np.asarray([
                        f"{p:.1f}%" if p > 0 else ""
                        for p in cm_normalized.flatten()
                    ]).reshape(global_confusion_matrix.shape)
    fig_cm=plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    # Heatmap zeichnen
    sns.heatmap(
        cm_normalized,
        annot=entries,
        fmt="",
        cmap="Blues",
        vmax=100,                     # Skala der Farbleiste fest auf 0-100% setzen
        vmin=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Recall per Class (%)'},
        annot_kws={"size": 11, "weight": "semibold"}
    )

    # Englische Titel und Achsenbeschriftungen
    plt.title('Overall Confusion Matrix (Normalized %)', fontsize=14, pad=15, weight='bold')
    plt.ylabel('Annotated Class', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Class', fontsize=12, labelpad=10)

    # Rotationen für perfekte Lesbarkeit
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Grafik anzeigen
    #plt.savefig('overall_confusion_matrix_percentage.png', dpi=300)
    plt.show()

    #Mean performance measures over the whole dataset (per class)
    for cls in range(num_classes):
        if cls in dice_per_class:
            mean_dice = np.mean(dice_per_class[cls])
            #mean_iou = np.mean(iou_per_class[cls])
            std_dice=np.std(dice_per_class[cls])

            #recheck script for 95% confidence intervall
            sem = stats.sem(dice_per_class[cls])
            n = len(dice_per_class[cls])
            ci_lower, ci_upper = stats.t.interval(0.95, df=n-1, loc=mean_dice, scale=sem)
            ci_lower=np.max([0.0, ci_lower])
            ci_upper=np.min([1.0, ci_upper])
            Dice_per_class.append(mean_dice)
            #IOU_per_class.append(mean_iou)
            message = f"Class {cls}: Mean Dice = {mean_dice:.4f}, STD_Dice = {std_dice:.4f}, ConfidenceIntervall_Dice (95%): [{ci_lower:.4f}, {ci_upper:.4f}], classname: {labels[cls]}"
            print(message)
            results_summary.append(message)

            ###other performance metrics averaged over the dataset
            mean_sensitivity = np.mean(sensitivity_per_class[cls])
            Sensitivity_per_class.append(mean_sensitivity)
            mean_precision = np.mean(precision_per_class[cls])
            Precision_per_class.append(mean_precision)
            mean_fnr =  np.mean(fnr_per_class[cls])
            Fnr_per_class.append(mean_fnr)
            message = f"Class {cls}: Mean Sensitivity = {mean_sensitivity:.4f}, Mean Precision= {mean_precision:.4f}, mean FNR = {mean_fnr:.4f}, classname: {labels[cls]}"
            print(message)
            results_summary.append(message)

        else: #does not happen in our 7 class EB wound dataset (all classes do occur)
            message = f"Class {cls}: Dice = N/A, IoU = N/A (not present),  classname: {labels[cls]}"
            print(message)
            results_summary.append(message)



    overall_dice = np.mean([np.mean(scores) for scores in dice_per_class.values()])
    overall_dice_wound_classes=  np.mean([np.mean(scores) for ind,scores in enumerate(dice_per_class.values()) if ind>=2]) ##first two classes are not wound classes
    np.mean([np.mean(scores) for scores in dice_per_class.values()])
    message = f"\n✅ Mean Dice (over all present classes): {overall_dice:.4f}, Mean Dice over wound classes (excluding class 0 and 1): {overall_dice_wound_classes:.4f}"
    print(message)
    results_summary.append(message)
    # overall_iou = np.mean([np.mean(scores) for scores in iou_per_class.values()])
    #message = f"✅ Mean IoU (over present classes): {overall_iou:.4f}"
    #print(message)
    #results_summary.append(message)
    overall_ACC = np.mean(ACC)
    message = f"✅ Mean ACC (percentage of correctly classified pixels): {overall_ACC:.4f}"
    print(message)
    results_summary.append(message)
    message = f"\n✅ Mean Dice (over present classes): {overall_dice:.4f}"

    overall_sensitivity_wound_classes=  np.mean([np.mean(scores) for ind,scores in enumerate(sensitivity_per_class.values()) if ind>=2])
    overall_precision_wound_classes=  np.mean([np.mean(scores) for ind,scores in enumerate(precision_per_class.values()) if ind>=2])
    overall_fnr_wound_classes=  np.mean([np.mean(scores) for ind,scores in enumerate(fnr_per_class.values()) if ind>=2])
    message =   f"\n✅  Mean Performance measures  over all wound classes (excluding class 0 and 1): Sensitivity:  {overall_sensitivity_wound_classes:.4f}, Precision:  {overall_precision_wound_classes:.4f}, FNR: {overall_fnr_wound_classes:.4f}"
    print(message)
    results_summary.append(message)
    # message=f"\n✅ Confusion Matrix:"
    # print(message)
    # results_summary.append(message)
    # results_summary.append(global_cm_normalized)


    return results_summary,fig_cm


def get_prediction_metrics_whole_dataset(mask_gt, mask_pred_root, labels, Imagename_is_Maskname=False):
    # ##Inputs:   path to the grountruth masks: mask_gt
    #             path to the predicted masks: mask_pred_root
    #             names of the classes_ labels
    # Output:     All results (Dice, IoU) per class and in total in the form of a string


    masks = glob.glob(mask_gt + '*.png')

    class_map = get_class_value_map(mask_gt)
    num_classes = len(class_map)

    ###this following part is only necessary if the masks and images have different names (Imagename_is_Maskname=False)
    masks.sort()
    mask_names = []
    for m in masks:
        mask_names.append(os.path.basename(m))
    mname = []
    annotator = []
    for m in mask_names:
        ind_last = m.rfind('_')
        #annotator.append(m[ind_last + 1:-4])
        mname.append(m[:ind_last])


    dice_per_class = defaultdict(list)
    iou_per_class = defaultdict(list)
    ACC=[]


    for ind,m in enumerate(masks):
        gt=cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        path_pred=glob.glob(os.path.join(mask_pred_root, mname[ind])+'*')
        path_pred=path_pred[0]
        if Imagename_is_Maskname:
            path_pred=os.path.join(mask_pred_root, os.path.basename(m))
        pred=cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        acc=np.sum(gt==pred)/np.sum(gt<10000)
        ACC.append(acc)

        for raw_val, class_idx in class_map.items():
            gt[gt == raw_val] = class_idx
            pred[pred == raw_val] = class_idx
        dice_scores, iou_scores = compute_metrics(pred, gt, num_classes)
        for cls, dice in dice_scores:
            dice_per_class[cls].append(dice)
        for cls, iou in iou_scores:
            iou_per_class[cls].append(iou)

    results_summary = []
    Dice_per_class = []
    IOU_per_class = []
    message = "\n--- Validation Metrics over the whole dataset (not seperated in folds) ---"
    print(message)
    results_summary.append(message)

    for cls in range(num_classes):
        if cls in dice_per_class:
            mean_dice = np.mean(dice_per_class[cls])
            mean_iou = np.mean(iou_per_class[cls])
            Dice_per_class.append(mean_dice)
            IOU_per_class.append(mean_iou)
            message = f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, classname: {labels[cls]}"
            print(message)
            results_summary.append(message)
        else:
            message = f"Class {cls}: Dice = N/A, IoU = N/A (not present),  classname: {labels[cls]}"
            print(message)
            results_summary.append(message)

    overall_dice = np.mean([np.mean(scores) for scores in dice_per_class.values()])
    overall_iou = np.mean([np.mean(scores) for scores in iou_per_class.values()])
    overall_ACC = np.mean(ACC)
    message = f"\n✅ Mean Dice (over present classes): {overall_dice:.4f}"
    print(message)
    results_summary.append(message)
    message = f"✅ Mean IoU (over present classes): {overall_iou:.4f}"
    print(message)
    results_summary.append(message)
    message = f"✅ Mean ACC (percentage of correctly classified pixels): {overall_ACC:.4f}"
    print(message)
    results_summary.append(message)

    return results_summary





def print_annotation_metrics_per_annotator(mask_root, labels):
    masks = glob.glob(mask_root + '*.png')

    masks.sort()
    mask_names = []
    for m in masks:
        mask_names.append(os.path.basename(m))

    mname = []
    annotator = []
    for m in mask_names:
        ind_last = m.rfind('_')
        annotator.append(m[ind_last + 1:-4])
        mname.append(m[:ind_last])

    mnames = list(set(mname))
    mnames.sort()

    annotators = list(set(annotator))
    annotator_scores = annotator_scores = {a: [] for a in annotators}
    class_map = get_class_value_map(mask_root)
    num_classes = len(class_map)

    dice_per_class = defaultdict(list)
    iou_per_class = defaultdict(list)
    ACC=[]
    dice_multi_class= []
    for m in mnames:
        indices = get_indices(m, mname)
        annos = [annotator[i] for i in indices]

        if len(indices) == 2:
            pred1 = cv2.imread(masks[indices[0]], cv2.IMREAD_GRAYSCALE)
            pred2 = cv2.imread(masks[indices[1]], cv2.IMREAD_GRAYSCALE)

            acc = np.sum(pred1 == pred2) / np.sum(pred1 < 10000)
            ACC.append(acc)

            for raw_val, class_idx in class_map.items():
                pred1[pred1 == raw_val] = class_idx
                pred2[pred2 == raw_val] = class_idx

            dice_scores, iou_scores = compute_metrics(pred1, pred2, num_classes)
            for cls, dice in dice_scores:
                dice_per_class[cls].append(dice)
                annotator_scores[annotator[indices[0]]].append(dice)
                annotator_scores[annotator[indices[1]]].append(dice)
            for cls, iou in iou_scores:
                iou_per_class[cls].append(iou)

            MCdice = 2 * np.sum(np.logical_and(pred1 == pred2, pred1 > 0)) / (np.sum(pred1 > 0) + np.sum(pred2 > 0))
            dice_multi_class.append(MCdice)

        elif len(indices) > 2:
            pairs = list(combinations(indices, 2))
            for p in pairs:
                pred1 = cv2.imread(masks[p[0]], cv2.IMREAD_GRAYSCALE)
                pred2 = cv2.imread(masks[p[1]], cv2.IMREAD_GRAYSCALE)

                acc = np.sum(pred1 == pred2) / np.sum(pred1 < 10000)
                ACC.append(acc)

                for raw_val, class_idx in class_map.items():
                    pred1[pred1 == raw_val] = class_idx
                    pred2[pred2 == raw_val] = class_idx

                dice_scores, iou_scores = compute_metrics(pred1, pred2, num_classes)
                for cls, dice in dice_scores:
                    dice_per_class[cls].append(dice)
                    annotator_scores[annotator[p[0]]].append(dice)
                    annotator_scores[annotator[p[1]]].append(dice)
                for cls, iou in iou_scores:
                    iou_per_class[cls].append(iou)

                MCdice = 2 * np.sum(np.logical_and(pred1 == pred2, pred1 > 0)) / (np.sum(pred1 > 0) + np.sum(pred2 > 0))
                dice_multi_class.append(MCdice)

    Dice_per_class = []
    IOU_per_class = []
    print("\n--- Validation Metrics ---")

    for anno in annotators:
        if len(annotator_scores[anno])>0:
            print('Average dice score of annotator {}:{} (nr of compared pairs:{})'.format(anno, np.mean(annotator_scores[anno]), len(annotator_scores[anno])))

    for cls in range(num_classes):
        if cls in dice_per_class:
            mean_dice = np.mean(dice_per_class[cls])
            mean_iou = np.mean(iou_per_class[cls])
            Dice_per_class.append(mean_dice)
            IOU_per_class.append(mean_iou)
            #print(f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, classname: {labels[cls]}")
            comparisons=len(dice_per_class[cls])
            print(f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, compared pairs ={comparisons}, classname: {labels[cls]}")
        else:
            print(f"Class {cls}: Dice = N/A, IoU = N/A (not present),  classname: {labels[cls]}")

    overall_dice = np.mean([np.mean(scores) for scores in dice_per_class.values()])
    overall_iou = np.mean([np.mean(scores) for scores in iou_per_class.values()])
    overall_ACC = np.mean(ACC)
    MultiClassDice=np.mean(dice_multi_class)
    print(f"\n✅ Mean Dice (over present classes): {overall_dice:.4f}")
    print(f"\n✅ Multi Class Dice Score: {MultiClassDice:.4f}")
    print(f"✅ Mean IoU (over present classes): {overall_iou:.4f}")
    print(f"✅ Mean ACC (percentage of correctly classified pixels): {overall_ACC:.4f}")



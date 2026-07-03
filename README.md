         
# EB-Segmentation-Mask2Former

Code to train and evaluate the Mask2Former architecture for Epidermolysis Bullosa (EB) wound image segmentation, developed for the paper **'AI-based segmentation of wound types caused by Epidermolysis Bullosa'**. 

*Note: A link to the paper will be provided here following its acceptance. Please cite the paper if you use this code in a publication.*

---

## Prerequisites & Installation

### Required Hardware
* **Tested Configuration:** NVIDIA GeForce RTX 3090 Ti (24GB VRAM)

### Required Python Packages
* `torch>=2.6.0+cu124` (Other versions should work, but this is the verified baseline)
* `torchvision>=0.21.0+cu124` (Other versions should work)
* Hugging Face `transformers` library
* `opencv-python` (`cv2`): Required **only** if you run full training combined with evaluation (`EB_mask2former.EB_seg`). It is **not** required for generating segmentation masks via inference alone (`EB_mask2former.EB_validate`).
* **Standard Scientific Libraries:** `numpy`, `pillow` (PIL), `matplotlib`, etc.

---

## Dataset Naming Convention
To ensure the scripts parse your data correctly, images and their corresponding ground truth masks must strictly follow this naming pattern:

* **Images (.jpg):** `Pat` + `PatientID` + `_` + `ImageID` + `_` + `AnnotatorID` + `.jpg`  
  * *Example:* `Pat01_a19A_annotatorA.jpg`
* **Masks (.png):** Must share the exact same filename prefix as the image but end in `.png`  
  * *Example:* `Pat01_a19A_annotatorA.png`


---

## Provided Trained Model
We provide a pre-trained weights file named **`mask2former_wound_segmentation_fold0.pth`**. This model was trained on the first fold (Fold 0) of the 4-fold cross-validation workflow described in our paper. 
---

## Usage Guide

To use these scripts, open your preferred Python environment (e.g., PyCharm, Jupyter Notebook, VS Code) and ensure your working directory is set to the folder containing this repository.

### 1. Apply Trained Model (Inference / Generating Masks)
Use this setup to segment wound regions in patient images using an already trained model. 

```python
import EB_mask2former

Param = {}
Param['OUTPUT_DIR'] = 'path/to/output/folder'         # Where segmentation masks will be written
Param['IMAGE_DIR'] = 'path/to/input/images'           # Folder containing your .jpg images to be segmented
Param['DEVICE'] = 'cuda:0'                            # Target GPU device (use 'cpu' if no GPU is available)
Param['NETDIR'] = 'path/to/mask2former_wound_segmentation_fold0.pth'         # Path to the provided trained model weights 

# Patch size configuration for sliding window inference:
# Images are processed patch-wise. If your source images are smaller than 2048x2048, 
# reduce this size to match your dataset constraints.
Param['INPUT_SIZE'] = (2048, 2048) 

# Run validation/inference
EB_mask2former.EB_validate(Param)
```

---

### 2. Train and Evaluate Your Own Model
Use this setup to run a full training loop alongside cross-validation evaluation.

```python
import EB_mask2former

Param = {}
Param['MASK_DIR'] = 'path/to/ground_truth/masks'      # Folder with manually segmented .png grayscale images
Param['IMAGE_DIR'] = 'path/to/wound/images'           # Folder with EB wound .jpg images
Param['OUTPUT_DIR'] = 'path/to/results/folder'        # Folder where metrics and masks will be saved
Param['DEVICE'] = 'cuda:0'                            # Target GPU device

# Training Optimization Parameters
Param['BATCH_SIZE'] = 2
Param['accumulation_steps'] = 2                      # Artificially doubles effective batch size to save GPU memory (set to 1 to disable)
Param['NR_FOLDS'] = 4                                 # Number of folds for k-fold cross-validation

# Class Settings
Param['LABEL_NAMES'] = ['Background,qual,creme', 'Unaffected', '(Re-)Epithelialized', 'Blister', 'Crust', 'Erosions', 'Tumor']

# Image Dimensions
Param['INPUT_SIZE'] = (1024, 1024)                    # Randomly cropped patch size extracted during training / baseline sliding window size for evaluation
Param['RESIZE'] = {'height': 512, 'width': 512}       # Internal resizing executed by the Mask2Former framework

# Execute pipeline
mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_mask2former.EB_seg(Param)
```

---

### 3. Advanced: Evaluation with Upscaled Patch Sizes (Sliding Window)
In our paper, we evaluated the Mask2Former architecture using a **larger patch size** for sliding window inference than what was used during the training phase. This technique improves final metrics while requiring significantly less GPU memory compared to training at that same high resolution.

To run this specific configuration, use the exact same setup script as the training section above, but append or modify these specific parameters:

```python
# Append these configurations to your evaluation block:
Param['eval_only'] = True
Param['NET_DIR'] = 'path/to/your/trained/models'
Param['INPUT_SIZE'] = (2048, 2048)                    # Increased default input patch size by a factor of 2
Param['RESIZE'] = {'height': 1024, 'width': 1024}     # Increased internal framework resizing by a factor of 2

# Execute inference evaluation
mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_mask2former.EB_seg(Param)
```

         
         
      
     

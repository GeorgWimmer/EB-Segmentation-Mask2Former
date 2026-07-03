# EB-Segmentation-Mask2Former
Code to train and evaluate the Mask2Former architecture for EB image segmentation for the paper 'AI-based segmentation of wound types caused by Epidermolysis Bullosa'. Link to the paper will be following after acceptance. Please cite the paper if you use the code in a publication.

# Needed python packages:
torch version 2.6.0+cu124, but others versions should work as well <br>
torchvision 0.21.0+cu124, but other versions should work as well <br>
Hugging Face's transformers library <br>
For training combined with evaluation of the model, the cv2 package is  needed (EB_seg in EB_mask2former.py), but not for generating segmentation masks from images (EB_validate in EB_mask2former.py) <br>
The other needed python libaries are standard libaries such as numpy, PIL, matplotlib,...<br>
We used the following GPU in our experients: NVIDIA GeForce RTX 3090 Ti with 24GB RAM <br>


# Apply trained Mask2Former to segment wound regions in images of EB patients
1) open an python editor (e.g. PyCharm, Jupyter Notebook,... )
2) go to the directory where you downloaded the code or import the directory
3) use the following code entry: <br>
    Param = {} <br>
    Param['OUTPUT_DIR'] =  'path where the segmentation masks are written out' <br>
    Param['IMAGE_DIR'] = 'path to the folder with the images to be segmented (jpg images)' <br>
    Param['INPUT_SIZE'] = (2048, 2048) # the image is processed patch-wise and the input size controls the patch size that is used for the sliding window inference. If the images are smaller than 2048 x 2048, then use a smaller input size for the patchwise segmentation of the image (sliding window inference) <br>
    Param['DEVICE'] = 'cuda:0' #The gpu device that will be used. If you have no gpu then use Param['DEVICE'] = 'cpu' <br>
    Param['NETDIR']= 'path to the trained model that is provided' (trained in fold 1 from the 4-fold cross validation in our paper)  <br>
    import EB_mask2former<br>
    EB_mask2former.EB_validate(Param)<br>

# Evaluate and/or train your own Mask2Former model to segment EB images
1) open an python editor (e.g. PyCharm, Jupyter Notebook,... )
2) go to the directory where you downloaded the code or import the directory
3) images and the corresponding ground truth masks must be named in the following way: Images:  'Pat' + PatientID + '_' + ImageID + '_' + AnnotatorID +.jpg (e.g.: 'Pat01_a19A_christian.jpg'). Masks: The same as the images but they have to be png images (e.g.: 'Pat01_a19A_christian.png') 
4) use the following code entry: <br>
    import EB_mask2former <br>
    Param={} <br>
    Param['MASK_DIR'] = 'path to the file with the manually segmented ground truth (png grayscale images)' <br>
    Param['OUTPUT_DIR']= 'path to the folder where the segmentations masks are written out as well as segmentation results' <br>
    Param['IMAGE_DIR'] = 'path to the folder with the EB wound images (jpg images)'  # <br>
    Param['DEVICE']='cuda:0' #The gpu device that will be used <br>
    Param['BATCH_SIZE'] = 2 <br>
    Param['accumulation_steps']=2 #to artificiall increase the batch size by this factor without needing more gpu memory  (if not needed then set to 1) <br>
    Param['NR_FOLDS']=4 # Number of folds for k-fold cross validation <br>
    Param['LABEL_NAMES']= ['Background,qual,creme', 'Unaffected', '(Re-)Epithelialized', 'Blister', 'Crust', 'Erosions', 'Tumor'] #Names of the different classes that are to be segmented <br>
    Param['INPUT_SIZE'] =(1024, 1024) # During training a patch of this size is extracted from  the image (after image augmentation)  at a random position and the patch is then fed to the model. During evaluation, this is the patch size used for sliding window inference <br>
    Param['RESIZE']={'height': 512, 'width': 512} # intern resizing done by  the Mask2Former framework. <br>
    mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_mask2former.EB_seg(Param) <br>

  6) In our paper, we evaluated the Mask2Former with a bigger input size (sliding window inference with bigger patch size) than used for training. This  increased the results and the gpu memory requirements for evaluation are smaller than for training. In this case, use the same code as for training and evaluation, but add the follwing parameters: <br>
         Param['eval_only']=True <br>
         Param['NET_DIR'] = 'path to the trained models' <br>
         Param['INPUT_SIZE'] = (2048, 2048) #e.g. increase the default input size by factor 2 <br>
         Param['RESIZE'] ={'height': 1024, 'width': 1024}  #e.g. increase the intern resizing done by  the Mask2Former framework by  factor 2 <br>



         
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
  * *Example:* `Pat01_a19A_christian.jpg`
* **Masks (.png):** Must share the exact same filename prefix as the image but end in `.png`  
  * *Example:* `Pat01_a19A_christian.png`

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
Param['NETDIR'] = 'path/to/trained_model.bin'         # Path to the provided trained model weights (e.g., Fold 1)

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
Param['INPUT_SIZE'] = (1024, 1024)                    # Randomly cropped patch size extracted during training / baseline sliding window size
Param['RESIZE'] = {'height': 512, 'width': 512}       # Internal resizing scale executed by the Mask2Former framework

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
Param['RESIZE'] = {'height': 1024, 'width': 1024}     # Increased internal framework scale by a factor of 2

# Execute inference evaluation
mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_mask2former.EB_seg(Param)
```

         
         
      
     

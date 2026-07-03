# EB-Segmentation-Mask2Former
Code to train and evaluate the Mask2Former architecture for EB image segmentation

# Needed python packages:
torch version 2.6.0+cu124, but others versions should work as well <br>
torchvision 0.21.0+cu124, but other versions should work as well <br>
Hugging Face's transformers library <br>
For training combined with evaluation of the model, the cv2 package is  needed (EB_seg in EB_mask2former.py), but not for generating segmentation masks from images (EB_validate in EB_mask2former.py) <br>
The other needed python libaries are standard libaries such as numpy, PIL, matplotlib,...

# Apply trained Mask2Former to segment wound regions in images of EB patients
1) open an python editor (e.g. PyCharm, Jupyter Notebook,... )
2) go to the directory where you downloaded the code or import the directory
3) use the following code entry:
    Param = {} <br>
    Param['OUTPUT_DIR'] =  'path where the segmentation masks are written out' <br>
    Param['IMAGE_DIR'] = 'path to the folder with the images to be segmented (jpg images)' <br>
    Param['INPUT_SIZE'] = (2048, 2048) #if the images are smaller than 2048 x 2048 then use a smaller input size for the patchwise segmentation of             the image (sliding window inference) <br>
    Param['RESIZE'] = {'height': 1024, 'width': 1024} # if you make the input size smaller, then also make this parameter smaller by the same factor <br>
    Param['DEVICE'] = 'cuda:0' #The gpu device that will be used. If you have no gpu then use Param['DEVICE'] = 'cpu'<br>
    import EB_mask2former<br>
    EB_mask2former.EB_validate(Param)<br>

# Evaluate and/or train your own Mask2Former model to segment EB images
1) open an python editor (e.g. PyCharm, Jupyter Notebook,... )
2) go to the directory where you downloaded the code or import the directory
3) images and the corresponding ground truth masks must be named in the following way: Images:  'Pat' + PatientID + '_' + ImageID + '_' + AnnotatorID +.jpg (e.g.: 'Pat01_a19A_christian.jpg'). Masks: The same as the images but they have to be png images (e.g.: 'Pat01_a19A_christian.png') 
4) use the following code entry:
    import EB_mask2former
    Param={}
    Param['MASK_DIR'] = 'path to the file with the manually segmented ground truth (png grayscale images)' <br>
    Param['OUTPUT_DIR']= 'path to the folder where the segmentations masks are written out as well as segmentation results' <br>
    Param['IMAGE_DIR'] = 'path to the folder with the EB wound images (jpg images)'  # <br>
    Param['DEVICE']='cuda:0' #The gpu device that will be used. If you have no gpu then use Param['DEVICE'] = 'cpu'<br>
    Param['BATCH_SIZE'] = 2
    Param['accumulation_steps']=2 #to artificiall increase the batch size by this factor without needing more gpu memory  (if not needed then set to 1) <br>
    Param['NR_FOLDS']=4 # Number of folds for k-fold cross validation <br>
    Param['LABEL_NAMES']= ['Background,qual,creme', 'Unaffected', '(Re-)Epithelialized', 'Blister', 'Crust', 'Erosions', 'Tumor'] #Nmes of the different classes that are to be segmented <br>
    mean_dice, mean_Dice_per_class, mean_iou, mean_IOU_per_class = EB_mask2former.EB_seg(Param) <br>

  5) Use additional parameters as described in the code (EB_seg in EB_mask2former)

# EB-Segmentation-Mask2Former
Code to train and evaluate the Mask2Former architecture for EB image segmentation

# Needed python packages:
torch version 2.6.0+cu124, but others versions should work as well
torchvision 0.21.0+cu124, but other versions should work as well
Hugging Face's transformers library
For training combined with evaluation of the model, the cv2 package is  needed (EB_seg in EB_mask2former.py), but not for generating segmentation masks from images (EB_validate in EB_mask2former.py)
The other needed python libaries are standard libaries such as numpy, PIL, matplotlib,...

# Apply trained Mask2Former to segment wound regions in images of EB patients
1) open an python editor (e.g. PyCharm, Jupyter Notebook,... )
2) go to the directory where you downloaded the code or import the directory
3) use the following code entry:

    Param['OUTPUT_DIR'] =  'path where the segmentation masks are written out'
    Param['IMAGE_DIR'] = 'path to the folder where the images are that should be segmented'
    Param['INPUT_SIZE'] = (2048, 2048) #if the images are smaller than 2048 x 2048 then use a smaller input size for the patchwise segmentation of the                             image (sliding window inference)
    Param['RESIZE'] = {'height': 1024, 'width': 1024} # if you make the input size smaller, then also make this parameter smaller by the same factor 
    Param['DEVICE'] = 'cuda:0' #The gpu device that will be used. If you have no gpu then use Param['DEVICE'] = 'cpu'
    import EB_mask2former
    EB_mask2former.EB_validate(Param)

# Evaluate and/or train your own Mask2Former model to segment EB images
1) open an python editor (e.g. PyCharm, Jupyter Notebook,... )
2) go to the directory where you downloaded the code or import the directory
3) use the following code entry:
  

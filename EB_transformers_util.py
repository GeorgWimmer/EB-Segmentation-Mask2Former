import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image, ImageEnhance
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from collections import defaultdict


def get_kfolds(foldnr, dataset):
    patients=dataset.patients
    pat=np.unique(patients)
    folds=np.array_split(pat,4)
    folds=[list(f) for f in folds]
    print(f'separation in {foldnr} folds:')
    print(folds)

    test_indices_per_fold=[]
    train_indices_per_fold=[]
    for k in range(0,foldnr):
        test_patients=folds[k]
        test_indices=[ind for ind, pat in enumerate(dataset.patients) if pat in test_patients]
        test_indices_per_fold.append(test_indices)
        train_indices=[ind for ind, pat in enumerate(dataset.patients) if pat not in test_patients]
        train_indices_per_fold.append(train_indices)

    return test_indices_per_fold, train_indices_per_fold



class WoundDataset(Dataset):
    def __init__(self, image_dir, mask_dir, class_map, transform=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], Imagename_is_Maskname=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.class_map = class_map
        #self.masks = list(sorted(os.listdir(mask_dir)))
        self.masks = sorted(
            f for f in os.listdir(mask_dir) if f.lower().endswith(".png")
        )
        self.normalize = TF.normalize
        self.mean = mean
        self.std = std
        self.Imagename_is_Maskname=Imagename_is_Maskname


        if self.Imagename_is_Maskname:
            self.images=sorted(
                f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")
            )
            #control if the images and masks are for the same images
            for ind,m in enumerate(self.masks):
                assert(m[:-4]==self.images[ind][:-4])
        else:
            #control if the images and masks are for the same images
            mm=[]
            for m in self.masks:
                mm.append(m[: m.rfind('_')])

            imgs = list(sorted(os.listdir(self.image_dir)))
            im_names=[im[:-4] for im in imgs]
            #im_ending=imgs[0][-4:]   #[im[-4:] for im in imgs]
            im_ending = [im[-4:] for im in imgs]
            relevant_images=[]
            for m in mm:
                if m in im_names:
                    ind=im_names.index(m)
                    ending=im_ending[ind]
                    #relevant_images.append(m+im_ending)
                    relevant_images.append(m+ending)
                else:
                    message=('Anonymized image not found for mask {}').format(m)
                    raise NameError(message)
            self.images=relevant_images

        patients=[]
        for m in self.masks:
            patients.append(m.split('_')[0])
        self.patients=patients


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assumes mask is grayscale with class indices

        if self.transform:
            image, mask = self.transform(image, mask)

        # # Map grayscale values to class indices
        # mask_np = np.array(mask)
        # mask_indices = np.zeros_like(mask_np, dtype=np.uint8)
        # for raw_val, class_idx in self.class_map.items():
        #     mask_indices[mask_np == raw_val] = class_idx

        # Ensure image is a tensor
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        image = self.normalize(image, mean=self.mean, std=self.std)

        # Convert mask to np array (grayscale values)
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask)
        elif isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            raise TypeError("Unsupported mask type")

        # Map grayscale values to class indices
        mask_indices = np.zeros_like(mask_np, dtype=np.uint8)
        for raw_val, class_idx in self.class_map.items():
            mask_indices[mask_np == raw_val] = class_idx

        mask = torch.from_numpy(mask_indices).long()

        return image, mask




class WoundDataset_mask2former(Dataset):
    def __init__(self, image_dir, mask_dir, processor, class_map, transform=None, ignore_index=255, Imagename_is_Maskname=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.class_map = class_map
        self.transform = transform
        #self.masks = list(sorted(os.listdir(mask_dir)))
        self.masks = sorted(
            f for f in os.listdir(mask_dir) if f.lower().endswith(".png")
        )
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.ignore_index=ignore_index
        self.Imagename_is_Maskname=Imagename_is_Maskname

        if self.Imagename_is_Maskname:
            self.images=sorted(
                f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")
            )
            #control if the images and masks are for the same images
            for ind,m in enumerate(self.masks):
                assert(m[:-4]==self.images[ind][:-4])
        else:
            #control if the images and masks are for the same images
            mm=[]
            for m in self.masks:
                mm.append(m[: m.rfind('_')])

            imgs = list(sorted(os.listdir(self.image_dir)))
            im_names=[im[:-4] for im in imgs]
            #im_ending=imgs[0][-4:]   #[im[-4:] for im in imgs]
            im_ending = [im[-4:] for im in imgs]
            relevant_images=[]
            for m in mm:
                if m in im_names:
                    ind=im_names.index(m)
                    ending=im_ending[ind]
                    #relevant_images.append(m+im_ending)
                    relevant_images.append(m+ending)
                else:
                    message=('Anonymized image not found for mask {}').format(m)
                    raise NameError(message)
            self.images=relevant_images

        patients=[]
        for m in self.masks:
            patients.append(m.split('_')[0])
        self.patients=patients


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assumes mask is grayscale with class indices

        if self.transform:
            image, mask = self.transform(image, mask)


        mask_np = np.array(mask)

        # mask_indices = np.zeros_like(mask_np, dtype=np.uint8)
        # for raw_val, class_idx in self.class_map.items():
        #     mask_indices[mask_np == raw_val] = class_idx
        # mask = Image.fromarray(mask_indices)
        # mask_np = np.array(mask)

        mask_indices = np.full_like(mask_np, fill_value=self.ignore_index, dtype=np.uint8)
        for raw_val, class_idx in self.class_map.items():
            if class_idx != self.ignore_index:
                mask_indices[mask_np == raw_val] = class_idx
        mask = Image.fromarray(mask_indices)
        mask_np = np.array(mask)

        # Map grayscale values to class indices


        class_labels = torch.unique(torch.tensor(mask_np)).tolist()  # list of present class indices

        # Create one-hot mask: shape [num_classes_in_image, H, W]
        mask_labels = []
        for c in class_labels:
            class_mask = (mask_np == c).astype(np.float32)
            mask_labels.append(torch.tensor(class_mask))
        mask_labels = torch.stack(mask_labels, dim=0)  # shape [C, H, W]

        # Processor
        inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt", ignore_index=self.ignore_index)
        pixel_values = inputs["pixel_values"].squeeze(0)
        pixel_mask = inputs["pixel_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "mask_labels": mask_labels,  # [num_present_classes, H, W]
            "class_labels": torch.tensor(class_labels),  # [num_present_classes]
        }




def collate_fn_mask2former(batch):
    collated = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "pixel_mask": torch.stack([item["pixel_mask"] for item in batch]),
        "mask_labels": [item["mask_labels"] for item in batch],        # keep as list
        "class_labels": [item["class_labels"] for item in batch],      # keep as list
    }
    return collated



# ----------------------------
# Dice Score for Multi-class
# ----------------------------
def dice_score(pred, target, num_classes):
    dice = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        score = (2 * intersection + 1e-7) / (union + 1e-7)
        dice.append(score.item())
    return dice


class AugmentationTransform: #returns normalized tensor images
    def __init__(self, crop_size=(1024, 1024), rotation=True, flip=True, resize=False, resizefactor=2, brightness=False,
                 contrast=False,
                 scaling=False,
                 min_mask_area=10000,
                 max_rotation=15):
        self.crop_size = crop_size
        self.rotation = rotation
        self.flip = flip
        self.resize = resize
        self.resizefactor = resizefactor
        self.brightness = brightness
        self.contrast = contrast
        self.scaling = scaling
        self.min_mask_area=min_mask_area
        self.max_rotation=max_rotation

    def __call__(self, image, mask):
        # Ensure both are PIL images
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        if isinstance(mask, torch.Tensor):
            mask = Image.fromarray(mask.numpy().astype(np.uint8))

        # Random Scaling
        if self.scaling:
            scale = random.uniform(0.8, 1.2)
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

        # Random Rotation
        if self.rotation:
            angle = random.uniform(- self.max_rotation, self.max_rotation)
            image = TF.rotate(image, angle, interpolation=Image.Resampling.BILINEAR, fill=(0,))
            mask = TF.rotate(mask, angle, interpolation=Image.Resampling.NEAREST, fill=(0,))

        # Random Horizontal Flip
        if self.flip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random Vertical Flip
        if self.flip and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        #Resize image (um generell grösseren Bereich im extrahierten  patch zu haben  wird das Bild kleiner gemacht --kleinere Auflösung, mehr inhalt im patch)
        if self.resize:
            im_size=image.size
            new_size=( np.int16(im_size[1] // self.resizefactor), np.int16(im_size[0] // self.resizefactor))
            image = TF.resize(image, new_size, interpolation=Image.Resampling.BILINEAR)
            mask = TF.resize(mask, new_size, interpolation=Image.Resampling.NEAREST)

        #Brightness Change(image only)
        if self.brightness:
            factor = random.uniform(0.8, 1.2)  # slight
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast Change (image only)
        if self.contrast:
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)

        # Convert mask to tensor to check values
        mask_tensor = TF.to_tensor(mask)[0]  # shape: [H, W]
        #non_bg_indices = (mask_tensor != 0).nonzero(as_tuple=False)

        for _ in range(10):  # Try multiple times to find a crop with non-bg
            i = random.randint(0, image.height - self.crop_size[0])
            j = random.randint(0, image.width - self.crop_size[1])

            mask_crop = mask_tensor[i:i+self.crop_size[0], j:j+self.crop_size[1]]
            non_bg_indices=mask_crop >0  ##changed recently--before false implementation
            if non_bg_indices.sum() > self.min_mask_area:
                image = TF.crop(image, i, j, self.crop_size[0], self.crop_size[1])
                mask = TF.crop(mask, i, j, self.crop_size[0], self.crop_size[1])
                break
        else:
            # fallback crop if all attempts are empty
            i = random.randint(0, image.height - self.crop_size[0])
            j = random.randint(0, image.width - self.crop_size[1])
            image = TF.crop(image, i, j, self.crop_size[0], self.crop_size[1])
            mask = TF.crop(mask, i, j, self.crop_size[0], self.crop_size[1])

        # Convert to tensor
        image = TF.to_tensor(image)  # [3, H, W], float32
        mask = torch.from_numpy(np.array(mask)).long()  # [H, W], int64

        return image, mask



class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, bias=False)
        self.gn = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class AugmentationTransform_mask2former:  #returns PIL images (unnormalized)
    def __init__(self, crop_size=(1024, 1024), rotation=True, flip=True, brightness=False,
                 contrast=False,
                 scaling=False, threshold=10000):
        self.crop_size = crop_size
        self.rotation = rotation
        self.flip = flip
        self.brightness = brightness
        self.contrast = contrast
        self.scaling = scaling
        self.threshold = threshold

    def __call__(self, image, mask):
        # Ensure both are PIL images
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        if isinstance(mask, torch.Tensor):
            mask = Image.fromarray(mask.numpy().astype(np.uint8))

        # Random Scaling
        if self.scaling:
            scale = random.uniform(0.8, 1.2)
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)

        # Random Rotation
        if self.rotation:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=Image.Resampling.BILINEAR, fill=(0,))
            mask = TF.rotate(mask, angle, interpolation=Image.Resampling.NEAREST, fill=(0,))

        # Random Horizontal Flip
        if self.flip and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random Vertical Flip
        if self.flip and random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        #Brightness Change(image only)
        if self.brightness:
            factor = random.uniform(0.8, 1.2)  # slight
            image = ImageEnhance.Brightness(image).enhance(factor)

        # Contrast Change (image only)
        if self.contrast:
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)

        w, h = image.size
        cw, ch = self.crop_size



        for _ in range(10):  # Try multiple times to find a crop with non-bg
            x = random.randint(0, w - cw)
            y = random.randint(0, h - ch)
            mask_crop = mask.crop((x, y, x + cw, y + ch))
            mask_np = np.array(mask_crop)
            if np.sum(mask_np>0)>self.threshold:
                mask=mask_crop
                image=image.crop((x, y, x + cw, y + ch))
                break
        else:
            x = random.randint(0, w - cw)
            y = random.randint(0, h - ch)
            mask=mask.crop((x, y, x + cw, y + ch))
            image=image.crop((x, y, x + cw, y + ch))



        return image, mask





def compute_class_weights(loader, num_classes):
    count = torch.zeros(num_classes)
    for _, masks in loader:
        for mask in masks:
            count += torch.bincount(mask.flatten(), minlength=num_classes)
    weight = 1.0 / (count + 1e-6)
    return (weight / weight.sum()) * num_classes




def get_class_value_map(mask_dir):
    """
    Scans all mask images and builds a mapping from unique grayscale values
    to class indices starting from 0.

    Returns:
        dict: grayscale_value -> class_index
    """
    unique_values = set()
    for fname in os.listdir(mask_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.tif')):
            continue
        mask_path = os.path.join(mask_dir, fname)
        mask = Image.open(mask_path).convert("L")
        values = np.unique(np.array(mask))
        unique_values.update(values.tolist())

    sorted_vals = sorted(unique_values)
    class_map = {v: i for i, v in enumerate(sorted_vals)}
    return class_map



def visualize_predictions(model, dataloader, class_colors, class_names=None, device='cuda:0', num_samples=3):
    """
    Visualizes model predictions from a dataloader batch.

    Args:
        model: Trained segmentation model.
        dataloader: A PyTorch DataLoader for validation or inference.
        class_colors: Dict mapping class ID to RGB tuple (e.g., {0: (0,0,0), 1: (255,0,0), ...})
        class_names: Optional list of class names for legend.
        device: 'cuda' or 'cpu'.
        num_samples: Number of batches/images to visualize.
    """
    model.eval()
    count = 0
    mean=dataloader.dataset.dataset.mean
    std=dataloader.dataset.dataset.std

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, dict) and 'logits' in outputs:
                outputs = outputs['logits']  # HuggingFace format

            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            for i in range(min(images.size(0), num_samples)):
                #image = images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, 3]
                image = denormalize_image(images[i].cpu(), mean=mean, std=std)
                gt_mask = masks[i].cpu().numpy()
                pred_mask = preds[i].cpu().numpy()

                def colorize_mask(mask):
                    h, w = mask.shape
                    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                    for cls_id, color in class_colors.items():
                        color_mask[mask == cls_id] = color
                    return color_mask

                gt_colored = colorize_mask(gt_mask)
                pred_colored = colorize_mask(pred_mask)

                # Plot
                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title("Input Image")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(gt_colored)
                plt.title("Ground Truth")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_colored)
                plt.title("Prediction")
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                count += 1
                if count >= num_samples:
                    return


#################################################
###inference
##################


def save_prediction_mask(mask, path, class_colors):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        rgb_mask[mask == class_id] = color
    Image.fromarray(rgb_mask).save(path)


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

        # # ✅ Skip only if class is not present in the ground truth --AUSGEBESSERT (also neue resultate werden schlechter sein)!!!!
        # if target_area == 0:
        #     continue
        # Skip if class doesn't appear in both pred and target
        if target_area == 0 and pred_area == 0:
            continue

        dice = (2. * intersection) / (pred_area + target_area + 1e-8)
        iou = intersection / (union + 1e-8)

        dice_scores.append((cls, dice))
        ious.append((cls, iou))

    return dice_scores, ious


def compute_metrics_binary(pred, target):
    dice_scores = []
    ious = []

    cls=1
    target_inds = (target == cls)
    pred_inds = (pred == cls)

    target_area = target_inds.sum()
    pred_area = pred_inds.sum()
    intersection = np.logical_and(pred_inds, target_inds).sum()
    union = np.logical_or(pred_inds, target_inds).sum()

    dice = (2. * intersection) / (pred_area + target_area + 1e-8)
    iou = intersection / (union + 1e-8)

    dice_scores.append((cls, dice))
    ious.append((cls, iou))

    return dice_scores, ious

######infereence SegFormer


def write_masks_from_val_dataset(val_dataset, model, output_dir, class_colors, device="cuda:0", num_classes=6,
                                 crop_size=(1024, 1024), stride=(896, 896), foldnr=0, label_names='', resize=False,
                                 resizefactor=2, sigmoid=False):

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "results.txt")

    if len(label_names)==0:
        label_names=list(range(0,num_classes))

    mean=val_dataset.dataset.mean
    std=val_dataset.dataset.std
    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std),])
    model = model.to(device)
    model.eval()
    dice_per_class = defaultdict(list)
    iou_per_class = defaultdict(list)

    for idx in tqdm(range(len(val_dataset)), desc="Generating full-size masks"):
        path_im = val_dataset.dataset.images[val_dataset.indices[idx]]
        path=os.path.join(val_dataset.dataset.image_dir, path_im)
        path_mask=val_dataset.dataset.masks[val_dataset.indices[idx]]

        mask=Image.open(os.path.join(val_dataset.dataset.mask_dir, path_mask))
        mask_np = np.array(mask)
        mask_indices = np.zeros_like(mask_np, dtype=np.uint8)
        class_map=val_dataset.dataset.class_map
        for raw_val, class_idx in class_map.items():
            mask_indices[mask_np == raw_val] = class_idx
        gt_mask = np.array(mask_indices)

        # Load full image again to avoid cropped input
        original_image = Image.open(path).convert("RGB")
        if resize: #make image smaller
            im_size=original_image.size
            new_size=( np.int16(im_size[1] // resizefactor), np.int16(im_size[0] // resizefactor) )
            original_image = TF.resize(original_image, new_size, interpolation=Image.Resampling.BILINEAR)

        pred_mask, _ = sliding_window_inference_blend_mask(
            image=original_image,
            model=model,
            transform=transform,
            crop_size=crop_size,
            stride=stride,
            num_classes=num_classes,
            device=device,
            sigmoid=False #nicht notwendig gewesen für EB
        )
        if resize:  #restore original size
            pred_mask_pil=Image.fromarray(pred_mask)
            original_size=(im_size[1], im_size[0])
            pred_mask_pil = TF.resize(pred_mask_pil, original_size, interpolation=Image.Resampling.NEAREST)
            pred_mask=np.asarray(pred_mask_pil)


        save_path = os.path.join(output_dir,path_im)
        if save_path[-3:] != 'png':  #use  png extension to avoid roundings (then masks are not usable beside of a visual impression)
            save_path=save_path[:-3]+'png'

        save_prediction_mask(pred_mask, save_path, class_colors)


        if sigmoid: #binary for krakelee project
            dice_scores, iou_scores = compute_metrics_binary(pred_mask, gt_mask)
        else:
            dice_scores, iou_scores = compute_metrics(pred_mask, gt_mask, num_classes)
        for cls, dice in dice_scores:
            dice_per_class[cls].append(dice)
        for cls, iou in iou_scores:
            iou_per_class[cls].append(iou)

    print(f"✅ Masks saved in  {output_dir}")

    results_summary = []
    Dice_per_class=[]
    IOU_per_class=[]
    print("\n--- Validation Metrics ---")
    for cls in range(num_classes):
        if cls in dice_per_class:
            mean_dice = np.mean(dice_per_class[cls])
            mean_iou = np.mean(iou_per_class[cls])
            Dice_per_class.append(mean_dice )
            IOU_per_class.append(mean_iou)
            print(f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, classname: {label_names[cls]}")
            results_summary.append(f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, classname: {label_names[cls]}")
        else:
            print(f"Class {cls}: Dice = N/A, IoU = N/A (not present), classname: {label_names[cls]}")
            results_summary.append(f"Class {cls}: Dice = N/A, IoU = N/A (not present), classname: {label_names[cls]}")
            Dice_per_class.append(1.0)
            IOU_per_class.append(1.0)

    overall_dice = np.mean([np.mean(scores) for scores in dice_per_class.values()])
    overall_iou = np.mean([np.mean(scores) for scores in iou_per_class.values()])
    print(f"\n✅ Mean Dice (over present classes): {overall_dice:.4f}")
    print(f"✅ Mean IoU (over present classes): {overall_iou:.4f}")
    results_summary.append(f"\nMean Dice (present classes): {overall_dice:.4f}")
    results_summary.append(f"Mean IoU  (present classes): {overall_iou:.4f}")

    # Write only final summary to file
    with open(log_file, "a") as log:
        log.write(f"--- Validation Summary for fold {foldnr} ---\n")
        for line in results_summary:
            log.write(line + "\n")

    return overall_dice, overall_iou, Dice_per_class, IOU_per_class





def make_blend_mask_segformer(crop_size):
    wy = np.hanning(crop_size)
    wx = np.hanning(crop_size)
    mask = np.outer(wy, wx)
    return np.float32(mask / mask.max())

###inference with blend mask to avoid artifacts
def sliding_window_inference_blend_mask(image, model, transform, crop_size=(1024, 1024), stride=(896, 896),
                                        num_classes=6, device="cuda:0", sigmoid=False):
    model.eval()
    assert(crop_size[0]==crop_size[1])
    image = image.convert("RGB")
    w, h = image.size
    full_probs = np.zeros((num_classes, h, w), dtype=np.float32)
    weight_sum= np.zeros((h, w), dtype=np.float32)
    #count_predictions = np.zeros((h, w), dtype=np.float32)
    blend_mask = make_blend_mask_segformer(crop_size[0])


    for y in range(0, h, stride[1]):
        for x in range(0, w, stride[0]):
            # x1, y1 = x, y
            # x2 = min(x1 + crop_size[0], w)
            # y2 = min(y1 + crop_size[1], h)
            x1 = min(x, w - crop_size[0])
            y1 = min(y, h - crop_size[1])
            x2, y2 = x1 + crop_size[0], y1 + crop_size[1]

            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(crop_tensor)
                # Überprüfen, ob das Attribut 'logits' existiert
                if hasattr(output, "logits"):
                    logits = output.logits
                else:
                    logits = output  # Es ist bereits der Tensor

                #logits = model(crop_tensor).logits

                if sigmoid: #binary classification
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)
                probs = F.interpolate(probs, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
                probs = probs.squeeze(0).cpu().numpy()
                #logits = F.interpolate(logits, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)



            #full_probs[:, y1:y2, x1:x2] += probs
            mask = blend_mask[:probs.shape[1], :probs.shape[2]]
            full_probs[:, y1:y2, x1:x2] += probs * mask
            weight_sum[y1:y2, x1:x2] +=  mask
            #count_predictions[y1:y2, x1:x2] += 1

    #full_probs /= np.maximum(count_predictions, 1e-5)
    #full_probs /= np.maximum(weight_sum, 1e-5)
    full_probs /= (weight_sum + 1e-8)
    if sigmoid:
        # print(full_probs.shape)
        # full_probs=np.squeeze(full_probs,axis=0)
        # print(full_probs.shape)
        #print(np.min(full_probs))
        #print(np.max(full_probs))
        prediction = np.squeeze(full_probs, axis=0)>0.5
        prediction = prediction.astype(np.uint8)
    else:
        prediction = np.argmax(full_probs, axis=0).astype(np.uint8)
    #return prediction, full_probs[2, :, :]
    return prediction, full_probs[-1, :, :] # corrected so that it also fits for the 2-class case for craquelure segmentations

##old inference
def sliding_window_inference(image, model, transform, crop_size=(1024, 1024), stride=(896, 896), num_classes=6, device="cuda:0"):
    model.eval()
    image = image.convert("RGB")
    w, h = image.size
    full_probs = np.zeros((num_classes, h, w), dtype=np.float32)
    #full_probs = da.zeros((num_classes, h, w), dtype=np.float32)
    count_predictions = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride[1]):
        for x in range(0, w, stride[0]):
            x1, y1 = x, y
            x2 = min(x1 + crop_size[0], w)
            y2 = min(y1 + crop_size[1], h)

            ##hinzugefügt damit die cropsize immer gleich bleibt.
            if x1 + crop_size[0]>w:
                x1=w-crop_size[0]
                x2=w
            if y1 + crop_size[1]>h:
                y1=h-crop_size[1]
                y2=h

            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = transform(crop).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(crop_tensor).logits
                logits = F.interpolate(logits, size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
                probs = logits.squeeze(0).cpu().numpy()




            full_probs[:, y1:y2, x1:x2] += probs
            count_predictions[y1:y2, x1:x2] += 1

    full_probs /= np.maximum(count_predictions, 1e-5)
    prediction = np.argmax(full_probs, axis=0).astype(np.uint8)
    return prediction




##inference mask2former

def write_masks_from_val_dataset_mask2former(val_dataset, model, output_dir, processor, class_colors, device="cuda:0", num_classes=6,
                                 patch_size=1024, stride=896, foldnr=0, label_names=''):

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "results.txt")

    if len(label_names)==0:
        label_names=list(range(0,num_classes))

    model = model.to(device)
    model.eval()
    dice_per_class = defaultdict(list)
    iou_per_class = defaultdict(list)

    for idx in tqdm(range(len(val_dataset)), desc="Generating full-size masks"):
        path_im = val_dataset.dataset.images[val_dataset.indices[idx]]
        path=os.path.join(val_dataset.dataset.image_dir, path_im)
        path_mask=val_dataset.dataset.masks[val_dataset.indices[idx]]

        # _, gt_mask = val_dataset[idx]
        # gt_mask = np.array(gt_mask)

        mask=Image.open(os.path.join(val_dataset.dataset.mask_dir, path_mask))
        mask_np = np.array(mask)
        mask_indices = np.zeros_like(mask_np, dtype=np.uint8)
        class_map=val_dataset.dataset.class_map
        for raw_val, class_idx in class_map.items():
            mask_indices[mask_np == raw_val] = class_idx
        gt_mask = np.array(mask_indices)

        # Load full image again to avoid cropped input
        original_image = Image.open(path).convert("RGB")

        ##Old inference without blend masks
        # pred_mask = infer_large_image_dask_mask2former(
        #     image=original_image,
        #     model=model,
        #     processor=processor,
        #     device=device,
        #     patch_size=patch_size,
        #     stride=stride,
        # )

        ###new inference with blend masks
        pred_mask = infer_large_image_mask2former(
            image=original_image,
            model=model,
            processor=processor,
            device=device,
            patch_size=patch_size,
            stride=stride,
        )

        #base_name = os.path.basename(path)
        #save_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_pred.png")
        save_path = os.path.join(output_dir,path_im)
        if save_path[-3:] != 'png':  # use  png extension to avoid roundings (then masks are not usable beside of a visual impression)
            save_path = save_path[:-3] + 'png'

        save_prediction_mask(pred_mask, save_path, class_colors)

        dice_scores, iou_scores = compute_metrics(pred_mask, gt_mask, num_classes)
        for cls, dice in dice_scores:
            dice_per_class[cls].append(dice)
        for cls, iou in iou_scores:
            iou_per_class[cls].append(iou)

    print(f"✅ Masks saved in  {output_dir}")

    results_summary = []
    Dice_per_class=[]
    IOU_per_class=[]
    print("\n--- Validation Metrics ---")
    for cls in range(num_classes):
        if cls in dice_per_class:
            mean_dice = np.mean(dice_per_class[cls])
            mean_iou = np.mean(iou_per_class[cls])
            Dice_per_class.append(mean_dice )
            IOU_per_class.append(mean_iou)
            print(f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, classname: {label_names[cls]}")
            results_summary.append(f"Class {cls}: Dice = {mean_dice:.4f}, IoU = {mean_iou:.4f}, classname: {label_names[cls]}")
        else:
            print(f"Class {cls}: Dice = N/A, IoU = N/A (not present), classname: {label_names[cls]}")
            results_summary.append(f"Class {cls}: Dice = N/A, IoU = N/A (not present), classname: {label_names[cls]}")
            Dice_per_class.append(1.0)
            IOU_per_class.append(1.0)

    overall_dice = np.mean([np.mean(scores) for scores in dice_per_class.values()])
    overall_iou = np.mean([np.mean(scores) for scores in iou_per_class.values()])
    print(f"\n✅ Mean Dice (over present classes): {overall_dice:.4f}")
    print(f"✅ Mean IoU (over present classes): {overall_iou:.4f}")
    results_summary.append(f"\nMean Dice (present classes): {overall_dice:.4f}")
    results_summary.append(f"Mean IoU  (present classes): {overall_iou:.4f}")

    # Write only final summary to file
    with open(log_file, "a") as log:
        log.write(f"--- Validation Summary for fold {foldnr} ---\n")
        for line in results_summary:
            log.write(line + "\n")

    return overall_dice, overall_iou, Dice_per_class, IOU_per_class




def make_blend_mask(patch_size):
    wy = np.hanning(patch_size)
    wx = np.hanning(patch_size)
    mask = np.outer(wy, wx)
    return torch.from_numpy(mask / mask.max()).float()

def compute_semantic_logits_from_queries(outputs, num_classes):
    class_logits = outputs["class_queries_logits"]        # [B, Q, num_classes+1]
    mask_logits = outputs["masks_queries_logits"]         # [B, Q, H, W]
    class_probs = class_logits.softmax(dim=-1)[..., :-1]  # ohne Hintergrund
    mask_probs = mask_logits.sigmoid()
    sem_seg = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    return sem_seg  # [B, num_classes, H, W]

##inference mask2former with blend mask to avoid artifacts
def infer_large_image_mask2former(
    image: Image.Image,
    model,
    processor,
    device,
    patch_size=1024,
    stride=896,
):
    model.eval()
    width, height = image.size
    num_classes = model.config.num_labels
    blend_mask = make_blend_mask(patch_size).to(device)
    full_probs = torch.zeros((num_classes, height, width), device=device)
    weight_sum = torch.zeros((height, width), device=device)

    # for top in tqdm(range(0, height, stride), desc="Sliding-window inference"):
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + patch_size, height)
            right = min(left + patch_size, width)

            ##hinzugefügt damit die cropsize immer gleich bleibt.
            if top + patch_size > height:
                top=height-patch_size
                bottom=height
            if left + patch_size > width:
                left=width-patch_size
                right=width



            patch = image.crop((left, top, right, bottom))
            inputs = processor(images=patch, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = compute_semantic_logits_from_queries(outputs, num_classes)
                logits = F.interpolate(
                    logits,
                    size=(bottom - top, right - left),
                    mode="bicubic",
                    align_corners=False,
                )
                probs = F.softmax(logits, dim=1)[0]
                mask = blend_mask[:probs.shape[1], :probs.shape[2]]
                full_probs[:, top:bottom, left:right] += probs * mask
                weight_sum[top:bottom, left:right] += mask

    full_probs /= (weight_sum.unsqueeze(0) + 1e-8)
    pred_mask = torch.argmax(full_probs, dim=0).cpu().numpy().astype(np.uint8)
    return pred_mask








###combo_loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, num_classes):
        pred = torch.softmax(pred, dim=1)  # [B, C, H, W]
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(pred * target_one_hot, dims)
        union = torch.sum(pred + target_one_hot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self, weights, num_classes):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = DiceLoss()
        self.num_classes = num_classes

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target, self.num_classes)
        return ce_loss + dice_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        logpt = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-logpt)
        focal_loss = ((1 - pt) ** self.gamma) * logpt
        return focal_loss.mean()


class DiceLoss_FL(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(inputs * targets, dim=(2,3))
        union = torch.sum(inputs + targets, dim=(2,3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedDiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.dice_loss = DiceLoss_FL()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        return self.dice_weight * self.dice_loss(inputs, targets) + \
               self.focal_weight * self.focal_loss(inputs, targets)

#

class DiceLoss_ignore_background(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target, num_classes):
        # pred: (B, C, H, W)
        # target: (B, H, W)
        B, C, H, W = pred.shape

        pred = torch.softmax(pred, dim=1)

        # Create mask of valid pixels
        valid_mask = target != self.ignore_index  # (B, H, W)

        # One-hot encode target and apply valid mask
        target_one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        valid_mask = valid_mask.unsqueeze(1).float()  # (B, 1, H, W)

        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(pred * target_one_hot, dims)
        union = torch.sum(pred + target_one_hot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class ComboLoss_ignore_background(nn.Module):
    def __init__(self, weights, num_classes, ignore_index=0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
        self.dice = DiceLoss_ignore_background(ignore_index=ignore_index)
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target, self.num_classes)
        return ce_loss + dice_loss

###


def visualize_class_color_map(class_value_map, class_names=None, save_path=None):
    n_classes = len(class_value_map)
    fig, ax = plt.subplots(1, figsize=(8, n_classes * 0.6))

    # Erzeuge das Farbfeld
    color_patches = np.zeros((n_classes * 40, 200, 3), dtype=np.uint8)

    for idx, (class_id, color) in enumerate(class_value_map.items()):
        color_patches[idx*40:(idx+1)*40, :] = color
        label = class_names[class_id] if class_names and class_id in class_names else f"Klasse {class_id}"
        ax.text(210, idx*40 + 20, label, va='center', fontsize=12)

    ax.imshow(color_patches)
    ax.axis('off')
    plt.tight_layout()
    plt.title("Class-Colormap")

    # Bild speichern, wenn Pfad angegeben ist
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Klassen-Colormap gespeichert unter: {save_path}")

    plt.show()


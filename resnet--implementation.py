import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
import warnings
import time 

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd,
    ResizeWithPadOrCropd, RandFlipd, RandRotate90d, RandAffined,
    RandGaussianNoised, ToTensord, Compose
)
from monai.data import Dataset
from monai.networks.nets import resnet18
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Data Loading 

def load_data(data_dir, max_samples=10):
    classes = ['SagittalasNifti', 'MetopicsNifti', 'NormalNifti']
    data = []
    for label, cls in enumerate(classes):
        cls_folder = os.path.join(data_dir, cls) 
        for fname in os.listdir(cls_folder):
            if fname.endswith('.nii.gz'):
                data.append({"image": os.path.join(cls_folder, fname), "label": label})
    return data

# Transformations 
def get_transforms(train=True):
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0)),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(96, 96, 96)),
    ]
    if train:
        transforms += [
            RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5),
            RandAffined(keys=["image"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1)),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
            #ScaleIntensityd(keys=["image"], a_min=0, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        ]
    transforms += [ToTensord(keys=["image", "label"])]
    return Compose(transforms)



def split_dataset_stratified(data, train_frac=0.8, val_frac=0.1):
    labels = [item["label"] for item in data]

    # First split: train vs temp (val+test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=(1 - train_frac), stratify=labels, random_state=42
    )

    # Compute val and test fractions relative to remaining data
    relative_val_frac = val_frac / (val_frac + (1 - train_frac - val_frac))

    # Second split: val vs test
    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_labels, test_size=(1 - relative_val_frac),
        stratify=temp_labels, random_state=42
    )

    return train_data, val_data, test_data


# Model Definition 

def get_model():
    return resnet18(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=3,
        pretrained=False
    )

    return model


# Training 

def train(data_dir, model_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(data_dir)
  
    train_data, val_data, test_data = split_dataset_stratified(data) # Split Dataset

    train_ds = Dataset(data=train_data, transform=get_transforms(train=True))
    val_ds = Dataset(data=val_data, transform=get_transforms(train=False))
    test_ds = Dataset(data=test_data, transform=get_transforms(train=False))


    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=4, pin_memory=True)

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0
    patience = 10
    no_improve_epochs = 0
    num_epochs = 50

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=["Sagittal", "Metopic", "Normal"]))
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_path)
            print("Saved best model")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping")
                break

    # Test
    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Test Accuracy:", accuracy_score(all_labels, all_preds))
    print("Test Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Sagittal", "Metopic", "Normal"]))

    return model, val_ds

from sklearn.metrics import precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1


# GradCAM
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_resized = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        return cam_resized.squeeze().cpu().numpy()

def apply_gradcam(model_path, val_ds, output_dir="gradcam_output-new"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Specify the target layer for Grad-CAM (the last convolutional layer of ResNet18)
    target_layer = model.layer4[1].conv2
    grad_cam = GradCAM3D(model, target_layer)

    # Class names for your dataset
    class_names = ["Sagittal", "Metopic", "Normal"]
    
    # Keep track of which classes we've processed
    seen_classes = set()

    # Loop through the validation dataset
    for i, sample in enumerate(val_ds):
        label = sample["label"]
        
      
        if label in seen_classes:
            continue
        
        # Add the current class to the set of seen classes
        seen_classes.add(label)

        # just added this 
        print(f"Original CT scan file used: {sample['image'].meta['filename_or_obj']}")

        # Process the image for Grad-CAM
        input_image = sample['image'].unsqueeze(0).to(device)  
        cam_output = grad_cam.generate_cam(input_image, class_idx=label)

        # Retrieve the original NIfTI image (to get dimensions)
        original_nii = nib.load(sample["image"].meta['filename_or_obj'])
        original_data = original_nii.get_fdata() 
        original_shape = original_data.shape  

        # Resize Grad-CAM output to match the original NIfTI image dimensions
        cam_output_resized = np.resize(cam_output, original_shape)

        # Save the Grad CAM image 
        nii_path = os.path.join(output_dir, f"gradcam_{class_names[label]}_{i}.nii.gz")
        nib.save(nib.Nifti1Image(cam_output, affine=np.eye(4)), nii_path)

        # Create and save the overlay slice image
        mid_slice = cam_output.shape[0] // 2
        cam_slice = cam_output[mid_slice]
        img_slice = input_image.cpu().numpy()[0, 0, mid_slice]  
        img_slice_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

        # Apply heatmap and overlay
        heatmap = cm.jet(cam_slice)[:, :, :3]
        overlay = 0.5 * heatmap + 0.5 * np.stack([img_slice_norm]*3, axis=-1)

        # Saves overlay images 
        fig, ax = plt.subplots(figsize=(6,6))  
        ax.imshow(overlay)
        plt.axis('off')
        plt.tight_layout()

        # Save the overlay with the exact dimensions as the original NIfTI file
        overlay_path = os.path.join(output_dir, f"gradcam_overlay_{class_names[label]}_{i}.png")
        plt.savefig(overlay_path, bbox_inches='tight')
        plt.close()

        print(f"Saved Grad-CAM for class {class_names[label]} to {overlay_path}")

       
        if len(seen_classes) == 3:
            break



if __name__ == "__main__":
    data_dir = "/home/svandana/scratch/cranio-project/data-copy-bal" 
    model_path = "best_model.pth"

    # Train and evaluate the model
    model, val_ds = train(data_dir, model_path=model_path)

    # Generate Grad-CAM visualization
    apply_gradcam(model_path, val_ds, output_dir="gradcam_output_check")


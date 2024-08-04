# GROUP19_COMP0197
import os
import random
import time
import glob

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from utils.paths import animals_10_dir, imagenet_dir


class Reduced_ImageNetDataset(Dataset):
    """
    Loads and cleans up images in Animals-10 dataset
    """
    def __init__(self, root_dir, target_size=(224, 224)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.target_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
            ])

        # Get all paths in the root directory
        all_paths = [os.path.join(root_dir, f) for f in os.listdir(self.root_dir)]

        # Filter out directories, keep only files
        self.images = sorted([f for f in all_paths if os.path.isfile(f) and f.endswith('.png')])

        # If code above doesn't find the files, the code below will.
        # Assume original dir structure i.e. `raw-img/**/<images files>`
        if self.images is None or self.images == []:
            img_dir = os.path.join(root_dir, '**')
            accepted_files = []
            accepted_fs = [os.path.join(img_dir, '*.jpg'),
                              os.path.join(img_dir, '*.jpeg'),
                              os.path.join(img_dir, '*.png')]
            for accepted_f in accepted_fs:
                accepted_files.extend(glob.glob(accepted_f, recursive=False))
            self.images = accepted_files

        print(f"Number of images: {len(self.images)}")  # Add this line

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        # Ensure image has 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(self.target_size)  # Resize the image
        if self.transform:
            img = self.transform(img)

        return img


class Animals10Dataset(Dataset):
    """
    Loads and cleans up images in Animals-10 dataset
    """
    def __init__(self, root_dir, target_size=(224, 224)):
        self.root_dir = root_dir
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.target_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
            ])

        # Get all paths in the root directory
        all_paths = [os.path.join(root_dir, f) for f in os.listdir(self.root_dir)]

        # Filter out directories, keep only files
        self.images = sorted([f for f in all_paths if os.path.isfile(f) and f.endswith('.png')])

        # If code above doesn't find the files, the code below will.
        # Assume original dir structure i.e. `raw-img/**/<images files>`
        if self.images is None or self.images == []:
            img_dir = os.path.join(root_dir, '**')
            accepted_files = []
            accepted_fs = [os.path.join(img_dir, '*.jpg'),
                              os.path.join(img_dir, '*.jpeg'),
                              os.path.join(img_dir, '*.png')]
            for accepted_f in accepted_fs:
                accepted_files.extend(glob.glob(accepted_f, recursive=False))
            self.images = accepted_files

        print(f"Number of images: {len(self.images)}")  # Add this line

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        # Ensure image has 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(self.target_size)  # Resize the image
        if self.transform:
            img = self.transform(img)

        return img


class PatchMasker:
    """
    Divide the image into non-overlapping patches and randomly mask a fixed number of patches.
    Args:
        image (torch.Tensor): The input color image represented as a PyTorch tensor with shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: The masked image.
    """
    def __init__(self, patch_size=32, num_patches_to_mask=5):
        self.patch_size = patch_size
        self.num_patches_to_mask = num_patches_to_mask

    def mask_patches(self, image):
        batch_size, channels, height, width = image.size()

        # Determine the number of patches in height and width
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # Create a list of all patch indices
        patch_indices = [(i, j) for i in range(num_patches_h) for j in range(num_patches_w)]

        # Randomly choose non-repeating patch indices to mask
        masked_patch_indices = random.sample(patch_indices, self.num_patches_to_mask)

        # Apply the mask to the image
        masked_image = image.clone()
        masks = torch.ones_like(masked_image)
        for index in masked_patch_indices:
            i, j = index
            y_start = i * self.patch_size
            y_end = min((i + 1) * self.patch_size, height)
            x_start = j * self.patch_size
            x_end = min((j + 1) * self.patch_size, width)
            masked_image[:, :, y_start:y_end, x_start:x_end] = 0  # Set masked patch to zero
            masks[:, :, y_start:y_end, x_start:x_end] = 0

        return masked_image, masks

    def rich_test(self, model, loader, device):
        """
        Takes
            patch masker object, link to image file
        Displays
            original and patched image
        Returns
            input image, masked image tensor
        """
        images = next(iter(loader))
        images = images.to(device)
        masked_images, masks = self.mask_patches(images[0:1])

        model.eval()
        with torch.no_grad():
            infill_images = model(masked_images.to(device))
            inpainted_images = masked_images.to(device) + infill_images.to(device) * (masked_images.to(device) == 0).float()

        # Denormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        images = images * std + mean
        masked_images = masked_images * std + mean
        infill_images = infill_images * std + mean
        inpainted_images = inpainted_images * std + mean

        # Move tensors back to CPU for plotting
        images = images.cpu()
        masked_images = masked_images.cpu()
        infill_images = infill_images.cpu()
        inpainted_images = inpainted_images.cpu()

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        axes[0].axis('off')
        axes[0].set_title('Image')
        axes[0].imshow(images[0].permute(1, 2, 0))

        axes[1].set_title('Masked Image')
        axes[1].imshow(masked_images[0].permute(1, 2, 0))

        axes[2].set_title('Infill')
        axes[2].imshow(infill_images[0].permute(1, 2, 0))

        axes[3].set_title('Inpainted')
        axes[3].imshow(inpainted_images[0].permute(1, 2, 0))

        plt.tight_layout()
        return plt


    def test(self, model, loader, display, device, directory, plot_and_image_file_title: str):
        """
        Takes
            patch masker object, link to image file
        Displays
            original and patched image
        Returns
            input image, masked image tensor
        """
        images = next(iter(loader))
        num_images = min(images.size(0), 4)
        masked_images, masks = self.mask_patches(images)
        model.eval()
        with torch.no_grad():
            infill_images = torch.sigmoid(model(masked_images.to(device)))
            inpainted_images = masked_images.to(device) + infill_images.to(device) * (masked_images.to(device) == 0).float()

        if display is False:
            plt.ioff()
        fig, axes = plt.subplots(4, num_images, figsize=(12, 9))
        time.sleep(1)
        fig.suptitle(plot_and_image_file_title)
        for i in range(num_images):
            if num_images > 1:
                ax0 = axes[0, i]
                ax1 = axes[1, i]
                ax2 = axes[2, i]
                ax3 = axes[3, i]
            else:
                ax0 = axes[0]
                ax1 = axes[1]
                ax2 = axes[2]
                ax3 = axes[3]
            ax0.axis('off')
            ax0.set_title('Image')
            ax0.imshow(images[i].permute(1, 2, 0))
            ax0.set_title('Masked Image')
            ax1.imshow(masked_images[i].permute(1, 2, 0))
            ax2.set_title('Infill')
            ax2.imshow(infill_images[i].cpu().permute(1, 2, 0))
            ax3.set_title('Inpainted')
            ax3.imshow(inpainted_images[i].cpu().permute(1, 2, 0))
        plt.tight_layout()
        return plt


def compute_mean_and_std_for_animals10():
    # From the unnormalised Animals-10, this function calculates a mean of tensor([0.5178, 0.5003, 0.4127]) and
    # standard deviation of tensor([0.2684, 0.2635, 0.2807])

    animals_ds = Animals10Dataset(imagenet_dir)
    animals_dataloader = DataLoader(animals_ds, batch_size=512, shuffle=True)

    channels_sum, channels_sqd_sum, num_batches = 0, 0, 0

    for data in animals_dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqd_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqd_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def compute_loss(model,dataloader, patch_masker, pt_criterion, params, device):
    '''
    compute the average loss per batch across the dataloader
    '''

    validation_loss=0.0
    model.eval()
    with torch.no_grad():
        for i, input_images in enumerate(dataloader):
            input_images = input_images.to(device)
            # Add random masking to the input images
            masked_images, masks = patch_masker.mask_patches(input_images)
            # Forward pass & compute the loss
            logits = model(masked_images)
            # squash to 0-1 pixel values
            masked_outputs = torch.sigmoid(logits) * masks  # dont calculate loss for masked portion
            loss = pt_criterion(masked_outputs, masked_images) / (1.0 - params['mask_ratio'])  #  normalise to make losses comparable across different mask ratios
            validation_loss=validation_loss+loss
    return validation_loss.cpu().numpy()/len(dataloader)


if __name__ == '__main__':
    m, s = compute_mean_and_std_for_animals10()

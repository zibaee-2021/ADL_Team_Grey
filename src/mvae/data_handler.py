import os
import random
import time

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Animals10Dataset(Dataset):
    """
    Loads and cleans up images in Animals-10 dataset
    """
    def __init__(self, root_dir, target_size=(224, 224), transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.5, 0.55], std=[0.2, 0.2, 0.2])  #  normalising helps convergence
            ])

        # Get all paths in the root directory
        all_paths = [os.path.join(root_dir, f) for f in os.listdir(self.root_dir)]

        # Filter out directories, keep only files
        self.images = sorted([f for f in all_paths if os.path.isfile(f)])

        print(f"Number of images: {len(self.images)}")  # Add this line

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        # Ensure image has 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')


        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize(self.target_size)  # Resize the image
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


    def test(self, model, loader, display, params, device):
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
        with torch.no_grad():
            infill_images = model(masked_images.to(device))
            inpainted_images = masked_images.to(device) + infill_images.to(device) * (masked_images.to(device) == 0).float()

        if display is False:
            plt.ioff()
        fig, axes = plt.subplots(4, num_images, figsize=(12, 9))
        time.sleep(1)
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
        plt.show()
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig('masks' + date_str + '.png')
        plt.close()

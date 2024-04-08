import os
import random
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
        self.transform = transform

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
        for index in masked_patch_indices:
            i, j = index
            y_start = i * self.patch_size
            y_end = min((i + 1) * self.patch_size, height)
            x_start = j * self.patch_size
            x_end = min((j + 1) * self.patch_size, width)
            masked_image[:, :, y_start:y_end, x_start:x_end] = 0  # Set masked patch to zero

        return masked_image


def test_patch_masker(patch_masker, image_file):
    """
    Takes
        patch masker object, link to image file
    Displays
        original and patched image
    Returns
        input image, masked image tensor
    """
    image = Image.open(image_file)
    image = image.resize((224,224))
    transform = transforms.ToTensor()
    input_image = transform(image).unsqueeze(0)
    masked_image_tensor = patch_masker.mask_patches(input_image)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].axis('off')
    axes[1].axis('off')
    axes[0].imshow(input_image[0].numpy().transpose((1,2,0)))
    axes[1].imshow(masked_image_tensor[0].numpy().transpose((1,2,0)))
    plt.show()

    return input_image[0], masked_image_tensor


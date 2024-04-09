# External packages
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time

from src.utils.paths import fine_tuning_dir


def view_batch(data_loader):
    for batch in data_loader:
        images, labels = batch
        print("Batch size:", images.size(0))

        # Convert one-hot labels to image
        labels_tensor = one_hot_to_tensor(labels)
        # Plot each image in the batch
        for i in range(images.size(0)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(images[i].permute(1, 2, 0))  # Matplotlib expects image in (H, W, C) format
            axes[1].imshow(labels_tensor[i].unsqueeze(0).permute(1, 2, 0))
            plt.tight_layout()
            plt.show()
        break  # Exit th


def view_training(model, test_loader, device):
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        outputs = model(images)

        # images = (images.cpu().detach().numpy()*255).astype(np.uint8)
        # labels = (labels.cpu().detach().numpy()*255).astype(np.uint8)
        output_labels = one_hot_to_tensor(outputs.cpu().detach())
        fig, axes = plt.subplots(3, 4, figsize=(12, 8))
        for i in range(4):
            ax = axes[0, i]
            ax.axis('off')
            ax.imshow(images[i].cpu().permute(1, 2, 0))
            ax = axes[1, i]
            ax.axis('off')
            ax.imshow(labels[i][0])
            ax = axes[2, i]
            ax.axis('off')
            ax.imshow(output_labels[i].unsqueeze(0).permute(1, 2, 0))
        plt.tight_layout()
        # TODO: move this logic
        #  if check_semantic_segmentation:
        plt.show()
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        plt.savefig(os.path.join(fine_tuning_dir, 'labels' + date_str + '.png'))
        break


def one_hot_to_tensor(one_hot):
    """
    Converts the one-hot segmented image produced by the model into a numpy array
    """
    if len(one_hot.shape) == 3:  # no batch dimension
        tensor = torch.argmax(one_hot, dim=0).float()
        tensor /= one_hot.size(0) - 1  #  normalise to [0,1]
    elif len(one_hot.shape) == 4:  # has batch dimension
        tensor = torch.argmax(one_hot, dim=1).float()
        tensor /= one_hot.size(1) - 1
    else:
        raise ValueError("Invalude input tensor shape")

    return tensor


class OxfordPetDataset(Dataset):
    """
    Takes
        links to jpg images and trimap pngs
    Returns
        image tensors and one-hot classification map
    """

    def __init__(self, image_dir, label_dir, parameters):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.parameters = parameters
        self.image_size = parameters['image_size']
        self.segment_classes = parameters['segmenter_classes']
        self.image_filenames = [filename for filename in os.listdir(image_dir) if filename.endswith('.jpg')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace('.jpg', '.png'))

        # read in image and convert to tensor
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor(), ])
        image_tensor = transform(image)

        # read in trimap, 1=foreground, 2=background, 3=indeterminate/boundary
        trimap = Image.open(label_path)
        label_transform = transforms.Compose([transforms.Resize((224, 224)),
                                              # Convert image to tensor # Scale pixel values to [0, 255] and convert to uint8
                                              transforms.ToTensor(),
                                              transforms.Lambda(lambda x: (x * 255).to(torch.uint8))])
        trimap_tensor = label_transform(trimap)

        # Create one-hot label encoding, including background and indeterminate
        segment_labels = torch.zeros((self.segment_classes,) + trimap_tensor.shape[1:], dtype=torch.int)
        segment_labels[0, :] = (trimap_tensor[0] == 1)  # foregrount
        segment_labels[1, :] = (trimap_tensor[0] == 2)  # background
        segment_labels[2, :] = (trimap_tensor[0] == 3)  # boundary

        return image_tensor, segment_labels

    def split_dataset(self, train_split, val_split, test_split, batch_size):
        """
        Reads in Oxford IIIT-pet images and splits in training, validation, test data loaders
        """
        # Load the dataset using ImageFolder
        # Calculate the sizes for train, validation, and test sets
        num_samples = len(self)
        num_train = int(train_split * num_samples)
        num_val = int(val_split * num_samples)
        num_test = num_samples - num_train - num_val

        # Shuffle indices and split into training, validation, and test sets
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Define samplers for each split
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        # Create data loaders for each set
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        val_loader = DataLoader(self, batch_size=4, sampler=val_sampler)
        test_loader = DataLoader(self, batch_size=4, sampler=test_sampler)

        return train_loader, val_loader, test_loader

## NEW

class OxfordPetDataset(Dataset):
    """
    Takes
        links to jpg images and trimap pngs
    Returns
        image tensors and classification map
    """

    def __init__(self, original_dataset, params):
        self.original_dataset = original_dataset
        self.params = params

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, target = self.original_dataset[idx]

        target_transform = transforms.Compose([
            transforms.Resize((self.params['image_size'], self.params['image_size'])),
            # Resize the input PIL Image to given size.
            transforms.ToTensor(),  # Convert a PIL Image to PyTorch tensor.
            transforms.Lambda(lambda x: ((x * 255) - 1).long())])  # Convert back to class values 0,1,2
        target = target_transform(target)

        return image, target

## NEW


class Animals10Dataset(Dataset):
    """
    Loads and cleans up images in Animals-10 dataset
    """

    def __init__(self, params):
        self.root_dir = params['script_dir'] + "/Data/Animals-10/raw-img"
        self.params = params,
        self.target_size = params['image_size']
        self.transform = transforms.Compose([
            transforms.Resize((self.target_size, self.target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.5, 0.55], std=[0.2, 0.2, 0.2])  #  normalising helps convergence
        ])
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            # Check if the class directory is a directory
            if not os.path.isdir(cls_dir):
                continue
            for filename in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, filename)
                images.append((img_path, self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        img = Image.open(img_path)

        # Ensure image has 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((self.target_size, self.target_size))
        # img = transforms.ToTensor()(img)

        img = img.resize((self.target_size, self.target_size))  # Resize the image
        if self.transform:
            img = self.transform(img)
        return img, label


def overlap(model, loader):
    images, labels = next(iter(loader))
    outputs = model(images.to(device))
    outputs = outputs.cpu().detach()
    images, labels = images.cpu(), labels.cpu()
    output_labels = torch.argmax(outputs.cpu().detach(), dim=1)
    overlap = labels == output_labels
    overlap_fraction = overlap.float().mean().item()

    return overlap_fraction

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

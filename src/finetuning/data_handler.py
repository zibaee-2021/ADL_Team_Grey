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

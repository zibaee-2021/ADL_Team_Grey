import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as v2T
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
from PIL import Image as Pil_Image
import os
import time
from skimage import io
import random
import numpy as np
import matplotlib.pyplot as plt


class OxfordPetDataset(Dataset):
    """
    Takes
        links to jpg images and trimap pngs
    Returns
        image tensors and one-hot classification map
    """

    def __init__(self, image_dir, label_dir, parameters, aug_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.parameters = parameters
        self.image_size = parameters['image_size']
        self.segment_classes = parameters['segmenter_classes']
        self.image_filenames = [filename for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace('.jpg', '.png'))

        # read in image and convert to tensor
        pil_image = Pil_Image.open(image_path).convert('RGB')
        # transform = T.Compose([T.Resize((self.image_size, self.image_size)),
        #                        T.ToTensor()])

        # SWITCHED TO THIS, ACCORDING TO PYTORCH DOCS RECOMMENDATIONS ON USING V2 FOR PERFORMANCE BOOST
        transform = v2T.Compose([
            v2T.ToImage(),  # this converts to a tv image TENSOR
            v2T.ToDtype(torch.uint8, scale=True),
            v2T.Resize(size=(self.image_size, self.image_size), antialias=True)
        ])

        image_tensor = transform(pil_image)
        # image_tensor = torch.rand(3,224,224)

        # ######### HERE I'VE INJECTED THE AUGMENTATION TRANSFORM ###########
        if self.aug_transform is not None:
            # image_ = io.imread(image_path)
            image_tensor = self.aug_transform(image_tensor)

        # read in trimap, 1=foreground, 2=background, 3=indeterminate/boundary
        trimap = Pil_Image.open(label_path)
        label_transform = v2T.Compose([
            # Convert image to tensor
            v2T.Resize((self.image_size, self.image_size), antialias=True),
            v2T.ToImage(),  # T.ToTensor() may be deprecated
            # Scale pixel values to [0, 255] and convert to uint8
            # T.Lambda(lambda x: (x * 255).to(torch.uint8))])
            v2T.Lambda(lambda x: (x * 255)),
            v2T.ToDtype(torch.uint8, scale=True)
        ])
        trimap_tensor = label_transform(trimap)

        # Create one-hot label encoding, including background and indeterminate
        segment_labels = torch.zeros((self.segment_classes,) + trimap_tensor.shape[1:], dtype=torch.int)
        segment_labels[0, :] = (trimap_tensor[0] == 2)  # background
        segment_labels[1, :] = (trimap_tensor[0] == 3)  # boundary
        segment_labels[2, :] = (trimap_tensor[0] == 1)  # foreground

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

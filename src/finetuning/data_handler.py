# GROUP19_COMP0197
# External packages
import os
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from matplotlib import pyplot as plt

from src.utils.paths import fine_tuning_dir

#
# def view_batch(data_loader):
#     for batch in data_loader:
#         images, labels = batch
#         print("Batch size:", images.size(0))
#
#         # Convert one-hot labels to image
#         labels_tensor = one_hot_to_tensor(labels)
#         # Plot each image in the batch
#         for i in range(images.size(0)):
#             fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#             axes[0].imshow(images[i].permute(1, 2, 0))  # Matplotlib expects image in (H, W, C) format
#             axes[1].imshow(labels_tensor[i].unsqueeze(0).permute(1, 2, 0))
#             plt.tight_layout()
#             plt.show()
#         break  # Exit th



def one_hot_to_tensor(one_hot):
    """
    Converts the one-hot segmented image produced by the model into a numpy array
    """
    if len(one_hot.shape) == 3:  # no batch dimension
        tensor = torch.argmax(one_hot, dim=0).float()
        tensor /= one_hot.size(0) - 1  # normalise to [0,1]
    elif len(one_hot.shape) == 4:  # has batch dimension
        tensor = torch.argmax(one_hot, dim=1).float()
        tensor /= one_hot.size(1) - 1
    else:
        raise ValueError("Invalude input tensor shape")

    return tensor


# class OxfordPetDataset(Dataset):
#     """
#     Takes
#         links to jpg images and trimap pngs
#     Returns
#         image tensors and one-hot classification map
#     """
#
#     def __init__(self, image_dir, label_dir, parameters, transform=None):
#         self.image_dir = image_dir
#         self.label_dir = label_dir
#         self.parameters = parameters
#         self.image_size = parameters['image_size']
#         self.segment_classes = parameters['segmenter_classes']
#         self.image_filenames = [filename for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.45, 0.5, 0.55], std=[0.2, 0.2, 0.2]) #normalising helps convergence
#                                         ])
#
#     def __len__(self):
#         return len(self.image_filenames)
#
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_filenames[idx])
#         label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace('.jpg', '.png'))
#
#         # read in image and convert to tensor
#         image = Image.open(image_path).convert('RGB')
#         image_tensor = self.transform(image)
#
#         # read in trimap, 1=foreground, 2=background, 3=indeterminate/boundary
#         trimap = Image.open(label_path)
#         label_transform = transforms.Compose([transforms.Resize((224, 224)),
#                                               # Convert image to tensor # Scale pixel values to [0, 255] and convert to uint8
#                                               transforms.ToTensor(),
#                                               transforms.Lambda(lambda x: (x * 255).to(torch.uint8))])
#         trimap_tensor = label_transform(trimap)
#
#         # Create one-hot label encoding, including background and indeterminate
#         segment_labels = torch.zeros((self.segment_classes,) + trimap_tensor.shape[1:], dtype=torch.int)
#         segment_labels[0, :] = (trimap_tensor[0] == 1)  # foregrount
#         segment_labels[1, :] = (trimap_tensor[0] == 2)  # background
#         segment_labels[2, :] = (trimap_tensor[0] == 3)  # boundary
#
#         return image_tensor, segment_labels
#
#     def split_dataset(self, train_split, val_split, test_split, batch_size):
#         """
#         Reads in Oxford IIIT-pet images and splits in training, validation, test data loaders
#         """
#         # Load the dataset using ImageFolder
#         # Calculate the sizes for train, validation, and test sets
#         num_samples = len(self)
#         num_train = int(train_split * num_samples)
#         num_val = int(val_split * num_samples)
#         num_test = num_samples - num_train - num_val
#
#         # Shuffle indices and split into training, validation, and test sets
#         indices = torch.randperm(num_samples)
#         train_indices = indices[:num_train]
#         val_indices = indices[num_train:num_train + num_val]
#         test_indices = indices[num_train + num_val:]
#
#         # Define samplers for each split
#         train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
#         val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
#         test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
#
#         # Create data loaders for each set
#         train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler, drop_last=True)
#         val_loader = DataLoader(self, batch_size=4, sampler=val_sampler)
#         test_loader = DataLoader(self, batch_size=4, sampler=test_sampler)
#
#         return train_loader, val_loader, test_loader

def view_training(model, loader: DataLoader, display:bool, device: torch.device, plot_and_image_file_title:str):
    # Note: plot_and_image_file_title cannot contain things like / , " :
    images, labels = next(iter(loader))
    num_images = min(images.size(0),4)
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        outputs = torch.sigmoid(outputs.cpu().detach())
        images, labels = images.cpu(), labels.cpu()
        output_labels = torch.argmax(outputs.cpu().detach(), dim=1)

    if display is False:
        plt.ioff()

    fig, axes = plt.subplots(4, num_images, figsize=(3*num_images,8))
    fig.suptitle(plot_and_image_file_title)
    time.sleep(1)
    for i in range(num_images):
        if num_images>1:
            ax0 = axes[0,i]
            ax1 = axes[1,i]
            ax2 = axes[2,i]
            ax3 = axes[3,i]
        else:
            ax0 = axes[0]
            ax1 = axes[1]
            ax2 = axes[2]
            ax3 = axes[3]
        ax0.axis('off')
        ax0.set_title('Image')
        ax0.imshow(images[i].permute(1,2,0))
        ax1.axis('off')
        ax1.set_title('Label')
        ax1.imshow(labels[i].permute(1,2,0))
        ax2.axis('off')
        ax2.set_title('Output (prob)')
        ax2.imshow(outputs[i].permute(1,2,0))
        ax3.axis('off')
        ax3.set_title('Output (argmax)')
        ax3.imshow(output_labels[i])
    plt.tight_layout()
    date_str = time.strftime("%H.%M_%d-%m-%Y_", time.localtime(time.time()))
    plt.savefig(os.path.join(fine_tuning_dir, date_str + plot_and_image_file_title + '.png'))
    # plt.show()
    plt.close()


def overlap(model, loader, device):
    images, labels = next(iter(loader))
    outputs = model(images.to(device))
    outputs = outputs.cpu().detach()
    images, labels = images.cpu(), labels.cpu()
    output_labels = torch.argmax(outputs.cpu().detach(), dim=1)
    overlap = labels == output_labels
    overlap_fraction = overlap.float().mean().item()

    return overlap_fraction


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
            transforms.Resize((self.params['image_size'], self.params['image_size'])),  # Resize the input PIL Image to given size.
            transforms.ToTensor(),                                            # Convert a PIL Image to PyTorch tensor.
            transforms.Lambda(lambda x: ((x*255)-1).long())])                 # Convert back to class values 0,1,2
        target = target_transform(target)

        return image, target


# I calculated mean and std dev on a random 256 batch of the OxfordPetDataset:
# mean is tensor([0.4811, 0.4492, 0.3958])
# std is tensor([0.2645, 0.2596, 0.2681])
#
# If normalisation is beneficial for the transform then these are the values to use.


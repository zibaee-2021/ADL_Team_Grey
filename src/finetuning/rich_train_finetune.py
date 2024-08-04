# GROUP19_COMP0197
import sys
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, random_split
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from utils.model_init import initialise_weights
from utils.optimizer import get_optimizer
from utils.paths import *
from utils.device import get_optimal_device
from finetuning.data_handler import (
    view_training,
    OxfordPetDataset, 
    overlap
)
from shared_network_architectures.networks_pt import (
    get_network,
    SegmentModel
)
from utils.IoUMetric import IoULoss

with open('rich_config.json', 'r') as f:
    params = json.load(f)

# Custom Loss
def soft_dice_loss(output, target, epsilon=1e-6):
    numerator = 2. * torch.sum(output * target, dim=(-2, -1))
    denominator = torch.sum(output + target, dim=(-2, -1))
    return (numerator + epsilon) / (denominator + epsilon)
    # return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))

class SoftDiceLoss(nn.Module):
    # source https://github.com/Nacriema/Loss-Functions-For-Semantic-Segmentation/blob/master/loss/__init__.py
    def __init__(self, reduction='none', use_softmax=True):
        """
        Args:
            use_softmax: Set it to False when use the function for testing purpose
        """
        super(SoftDiceLoss, self).__init__()
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, output, target, epsilon=1e-6):
        """
        References:
        JeremyJordan's Implementation
        https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py

        Paper related to this function:
        Formula for binary segmentation case - A survey of loss functions for semantic segmentation
        https://arxiv.org/pdf/2006.14822.pdf

        Formula for multiclass segmentation cases - Segmentation of Head and Neck Organs at Risk Using CNN with Batch
        Dice Loss
        https://arxiv.org/pdf/1812.02427.pdf

        Args:
            output: Tensor shape (N, N_Class, H, W), torch.float
            target: Tensor shape (N, H, W)
            epsilon: Use this term to avoid undefined edge case

        Returns:

        """
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        if self.reduction == 'none':
            return 1.0 - soft_dice_loss(output, one_hot_target)
        elif self.reduction == 'mean':
            return 1.0 - torch.mean(soft_dice_loss(output, one_hot_target))
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


# Hyper-parameters
ft_batch_size = params["ft_batch_size"]
ft_num_epochs = params["ft_num_epochs"]
ft_lr = params["learning_rate"]
num_classes = params['num_classes']

#experiment_name = f"animals10_{params["network"]}_{params["dataset_name"]}_finetune"
experiment_name = f"imagenet_{params["network"]}_finetune"
wandb.init(name=experiment_name, project = "last_Day", entity="adl_team_grey", config=params)

oxford_path = oxford_3_dir

# Setup
device = get_optimal_device()
baseline = False

if __name__ == '__main__':

    encoder, decoder = get_network(params, num_classes)
    if baseline:
        encoder = initialise_weights(encoder)
    else:
        encoder_path = "imagenet_pretrained.pt"
        #encoder_path = "animals10_pretrained.pt"
        assert os.path.exists(encoder_path), \
            f"Could not find {encoder_path} in {models_dir}"
        state_dict = torch.load(encoder_path)
        encoder.load_state_dict(torch.load(encoder_path), strict=False)

    segment_model = SegmentModel(encoder, decoder).to(device)
    
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize((params['image_size'], params['image_size'])),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ])
    
    train_dataset = datasets.OxfordIIITPet(root=os.path.join(oxford_3_dir, "Train"),
                                           split='trainval',
                                           download=True,
                                           target_types='segmentation',
                                           transform=transform)
    test_dataset = datasets.OxfordIIITPet(root=os.path.join(oxford_3_dir, "Test"),
                                          split='test',
                                          download=True,
                                          target_types='segmentation',
                                          transform=transform)

    full_dataset = ConcatDataset([train_dataset, test_dataset])
    test_size = int(0.2 * len(full_dataset))  # 30% for testing
    trainval_size = len(full_dataset) - test_size

    trainval_dataset, test_dataset = random_split(full_dataset, [trainval_size, test_size])
    val_size = int(0.2 * len(trainval_dataset))  # 20% of the remaining 70% for validation
    train_size = len(trainval_dataset) - val_size
    train_dataset, val_dataset = random_split(trainval_dataset, [train_size, val_size])

    # Create the DataLoaders
    oxford_train_dataset = OxfordPetDataset(train_dataset, params)
    train_loader = torch.utils.data.DataLoader(oxford_train_dataset, batch_size=params['ft_batch_size'], shuffle=True)

    oxford_val_dataset = OxfordPetDataset(val_dataset, params)
    val_loader = torch.utils.data.DataLoader(oxford_val_dataset, batch_size=params['ft_batch_size'], shuffle=False)

    oxford_3_test_dataset = OxfordPetDataset(test_dataset, params)
    test_loader = torch.utils.data.DataLoader(oxford_3_test_dataset, batch_size=params['ft_batch_size'], shuffle=True)

    class_weights = torch.tensor(params['class_weights']).to(device)
    optimizer = get_optimizer(segment_model, params)

    loss_func_choice = {'cel': torch.nn.CrossEntropyLoss(weight=class_weights),
                            'mse': torch.nn.MSELoss(),
                            'bce': torch.nn.BCELoss(),
                            'iou': IoULoss(preds_are_logits=False).forward}
    
    #criterion = SoftDiceLoss(reduction='mean', use_softmax=True).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    preplot = view_training(segment_model, train_loader, True, device)
    wandb.log({"Before Training": preplot})
    print(f"Starting overlap: {overlap(segment_model, train_loader, device):.3f}")

    # Define early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(ft_num_epochs):

        segment_model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1)
            optimizer.zero_grad()
            outputs = segment_model(images)
            #loss = criterion.forward(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            wandb.log({"Running Loss": loss.item()})

        train_loss /= len(train_loader)
        wandb.log({"Epoch": epoch + 1, "Training Loss": train_loss})

        segment_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze(1)
                outputs = segment_model(images)
                loss = criterion.forward(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss})

        # Early stopping
        if val_loss < best_val_loss: # add condition for if pretrained (will want to do earlier)
            print(f'Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(segment_model.state_dict(), 'best_model.pt')
            best_val_loss = val_loss
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    postplot = view_training(segment_model, train_loader, True, device)
    wandb.log({"After Training Trainset Seg": postplot})

    # Load the best model
    segment_model.load_state_dict(torch.load('best_model.pt'))

    # Test
    segment_model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1)
            outputs = segment_model(images)
            loss = criterion.forward(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

    average_test_loss = total_loss / num_batches
    wandb.log({"Average Test Loss": average_test_loss})
    print(f"Final overlap: {overlap(segment_model, test_loader, device):.3f}")

    testplot = view_training(segment_model, test_loader, True, device)
    wandb.log({"Testset Seg": testplot})
    wandb.finish()

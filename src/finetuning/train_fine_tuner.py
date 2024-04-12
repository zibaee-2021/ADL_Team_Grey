# External packages
from datetime import datetime
import torch
from torchvision import datasets, transforms
import time
import random
import numpy as np
from matplotlib import pyplot as plt

# our code
from src.utils.model_init import initialise_weights
from src.utils.optimizer import get_optimizer
from src.utils.paths import *
from src.utils.device import get_optimal_device
from src.finetuning.data_handler import (
    view_training,
    OxfordPetDataset, overlap
)
from src.shared_network_architectures.networks_pt import (
    get_network,
    SegmentModel
)

from src.IoUMetric import IoULoss

# TODO:
"""
- add wandb for param checking
- tidy up config
- tidy up saving locations (losses/plots) further
- Think about saving models (with datestr in name)... do we always manually set fine tuner?
"""

## Control
## Training
check_oxford_batch = True
run_fine_tuning = True
pre_training_model_encoder = None  # set these to the pretrain models you want to use
pre_training_model_decoder = None
check_semantic_segmentation = True
save_models = run_fine_tuning
load_models = not run_fine_tuning
report_every = 25

## Testing
# TODO: again we need to think about loading models
# check_oxford_batch = True
# run_semantic_training = True
# check_semantic_segmentation = True
# save_models = run_semantic_training
# load_models = not run_semantic_training
# report_every = 25

## Definition s
## TODO: unify paramaterws for different models (a central config? for shared params?)
## Issue with that is when people want to try different paramaters...
params = {
    # Image
    "image_size": 224,  # number of pixels square
    "num_channels": 3,  #  RGB image -> 3 channels
    'num_classes': 3,

    # Network
    'network': "CNN",  # CNN, ViT, Linear
    'num_features': 768,  # 768
    'hidden_dim': 2048,
    "vit_num_layers": 4,  # 12ViT parameter
    "vit_num_heads": 8,  # 8 ViT parameter
    "vit_mlp_dim": 2048,  # 1024 ViT parameter#

    # hyper-parameters
    "ft_batch_size": 32,
    "learning_rate": 0.001,
    "momentum": 0.9,

    # Training
    'optimizer': "Adam",  # Adam, AdamW, SGD
    'ft_num_epochs': 8,
    'class_weights': [1.0, 0.5, 1.5],  #  pet, background, boundary
}

# Hyper-parameters
ft_batch_size = params["ft_batch_size"]
ft_num_epochs = params["ft_num_epochs"]
ft_lr = params["learning_rate"]

# file paths
oxford_path = oxford_3_dir

# test image
test_image_path = os.path.join(oxford_3_dir, "images/Abyssinian_1.jpg")

if __name__ == '__main__':
    # Set seeds for random number generator
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = get_optimal_device()

    # initialize model: encoder and decoder
    encoder, decoder = get_network(params, params['num_classes'])

    if pre_training_model_encoder:
        encoder_path = os.path.join(models_dir, pre_training_model_encoder)
        assert os.path.exists(encoder_path), \
            f"Could not find {pre_training_model_encoder} in {models_dir}"
        encoder.load_state_dict(torch.load(encoder_path), strict=False)
    else:
        print(f"Initialising encoder randomly")
        initialise_weights(encoder)
    if pre_training_model_decoder:
        decoder_path = os.path.join(models_dir, pre_training_model_decoder)
        assert os.path.exists(decoder_path), \
            f"Could not find {pre_training_model_decoder} in {models_dir}"
        decoder.load_state_dict(torch.load(decoder_path), strict=False)
    else:
        print(f"Initialising decoder randomly")
        initialise_weights(decoder)

    segment_model = SegmentModel(encoder, decoder).to(device)

    transform = transforms.Compose([transforms.Resize((params['image_size'], params['image_size'])),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.45, 0.5, 0.55], std=[0.2, 0.2, 0.2])
                                    # You can try these calculated mean and std dev:
                                    # mean is [0.4811, 0.4492, 0.3958]
                                    # std is [0.2645, 0.2596, 0.2681]

                                    #  normalising helps convergence
                                    ])  # Define data transformations: resize and convert to PyTorch tensors

    # Fetch dataset (if not already saved locally)
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

    # # Initialize dataLoader
    # TODO: do we want to split train into test / val?
    oxford_3_train_dataset = OxfordPetDataset(train_dataset, params)
    train_loader = torch.utils.data.DataLoader(oxford_3_train_dataset, batch_size=ft_batch_size, shuffle=True)
    oxford_3_test_dataloset = OxfordPetDataset(test_dataset, params)
    test_loader = torch.utils.data.DataLoader(oxford_3_test_dataloset, batch_size=ft_batch_size, shuffle=True)

    ############################
    if run_fine_tuning:
        print("In fine-tuning")
        start_time = time.perf_counter()

        # loss and optimiser
        class_weights = torch.tensor(params['class_weights']).to(device)
        # criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)  # TODO: This was here before. Remove?

        # Use of weights to correct for disparity in foreground, background, boundary
        loss_func_choice = {'cel': torch.nn.CrossEntropyLoss(weight=class_weights),
                            'mse': torch.nn.MSELoss(),
                            'bce': torch.nn.BCELoss(),
                            'iou': IoULoss(preds_are_logits=False).forward}

        criterion = loss_func_choice['cel']
        criterion = criterion.to(device)

        optimizer = get_optimizer(segment_model, params)

        # test everything is working
        print("View images, labels and as yet unlearned model output before starting")
        view_training(segment_model, train_loader, True, device)
        print(f"Starting overlap: {overlap(segment_model, train_loader, device):.3f}")

        # Training loop
        # TODO: add wandb & average loss
        losses = []
        for epoch in range(ft_num_epochs):
            epoch_start_time = time.perf_counter()
            running_loss = 0

            for its, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                labels = labels.squeeze(1)

                # forward + backward + optimize
                outputs = segment_model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.detach().cpu().item()
                if its % report_every == (report_every - 1):  # print every report_every mini-batches
                    curr_time = time.perf_counter() - start_time
                    print(
                        'Epoch [%d / %d],  %d image minibatch [%4d / %4d], cumulative running loss: %.4f, uptime: %.2f' % (
                            epoch + 1, ft_num_epochs, ft_batch_size, its + 1, len(train_loader),
                            running_loss / len(train_loader), curr_time))

            epoch_end_time = time.perf_counter()
            losses.append(running_loss / len(train_loader))
            print(
                f"Epoch [{epoch + 1}/{ft_num_epochs}] completed in {(epoch_end_time - epoch_start_time):.0f}s, Loss: {running_loss / len(train_loader):.4f}")
            #TODO: if we had validation, use validation here rather than test
            view_training(segment_model, test_loader, True, device)
        end_time = time.perf_counter()
        print(f"Segmentation training finished after {(end_time - start_time):.0f}s")

        # save the trained model and losses
        if save_models:
            print("Saving Models")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Current timestamp
            final_epoch_loss = losses[-1]  # Replace 'epoch_loss' with the actual value

            # Define model file names
            ft_model_file = f"ft_model_{timestamp}.pt"
            ft_encoder_model_file = f"ft_encoder_model_{timestamp}.pt"
            ft_decoder_model_file = f"ft_decoder_model_{timestamp}.pt"
            # ft_epoch_loss_file = f"ft_epoch_loss_{final_epoch_loss}.txt"  # Optionally include epoch loss in the file name

            encoder_path = os.path.join(models_dir, ft_encoder_model_file)
            torch.save(encoder.state_dict(), encoder_path)
            print(f"Saved {encoder_path}")

            decoder_path = os.path.join(models_dir, ft_decoder_model_file)
            torch.save(decoder.state_dict(), decoder_path)
            print(f"Saved {decoder_path}")

        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        with open(os.path.join(fine_tuning_dir, "ft_losses" + date_str + ".txt"), 'w') as f:
            for i, loss in enumerate(losses):
                f.write(f'{i}  {loss}\n')

        plt.plot(losses)
        plt.title("Fine-tuner losses")
        date_str = time.strftime("_%H.%M_%d-%m-%Y", time.localtime(time.time()))
        # TODO: think about saving
        plt.savefig('ft_losses' + date_str + '.png')
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.show()
        plt.close()
        view_training(segment_model, train_loader, True, device)  # dont understand why this doesnt display

    # display inference on test set
    if check_semantic_segmentation:
        for its in range(5):
            view_training(segment_model, test_loader, True, device)
            print(f"Sample test set overlap: {overlap(segment_model, test_loader, device):.3f}")

    print("Fine-tuning script complete")

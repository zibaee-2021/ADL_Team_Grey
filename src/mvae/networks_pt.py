import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MaskedAutoencoder(nn.Module):
    """
    Defines encoder / decoder for masked autoencoder pre-trainer
    Takes
        batch of images
    Returns
        image passed through encoder and decoder
    """

    def __init__(self, encoder, decoder):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VisionTransformerEncoder(nn.Module):
    """
    Defines vision transformer model
    Takes
        batch of images
    Returns
        vision-transformer feature embeddings of those images
    """

    def __init__(self, params):
        super(VisionTransformerEncoder, self).__init__()
        self.image_size = params['image_size']
        self.patch_size = params['patch_size']
        self.num_features = params['vit_num_features']
        self.num_layers = params['vit_num_layers']
        self.num_heads = params['vit_num_heads']
        self.hidden_dim = params['vit_hidden_dim']
        self.mlp_dim = params['vit_mlp_dim']
        self.vit = models.vision_transformer.VisionTransformer(image_size=self.image_size,
                                                               patch_size=self.patch_size,
                                                               num_classes=self.num_features,
                                                               num_layers=self.num_layers,
                                                               num_heads=self.num_heads,
                                                               hidden_dim=self.hidden_dim,
                                                               mlp_dim=self.mlp_dim)

    def forward(self, x):
        # Pass the input through the ViT-B_16 backbone
        features = self.vit(x)
        return features


# Define color image decoder
class VisionTransformerDecoder(nn.Module):
    """
    Decoder
    Takes
       batch of image feature embeddings
    Returns
        reconstructed image tensor: Batch * Channels (3) * Image size (224) * Image size (224)

    In the mvae process the three output dimensions are the three colours,
    in the segmentation process these get re-interpretted as probabilities
    over the three classes (foreground, background, boundary) - bit of a hack
    """

    def __init__(self, params):
        super(VisionTransformerDecoder, self).__init__()
        self.input_dim = params['vit_num_features']
        self.image_size = params['image_size']
        self.num_channels = params['num_channels']
        self.output_dim = self.image_size * self.image_size * self.num_channels
        self.hidden_dim_one = params['decoder_hidden_dim']
        self.CNN_channels = params['decoder_CNN_channels']
        self.upscale = params['decoder_scale_factor']
        self.CNN_patch = int(self.image_size / 2 ** 2 / self.upscale)
        self.hidden_dim_two = int(
            self.CNN_channels * self.CNN_patch ** 2)  # 5 upsampling layers, halving channels (except last layer)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_one)
        self.fc2 = nn.Linear(self.hidden_dim_one, self.CNN_channels * self.CNN_patch * self.CNN_patch)
        self.unflatten = nn.Unflatten(1, (self.CNN_channels, self.CNN_patch, self.CNN_patch))
        self.conv1 = nn.ConvTranspose2d(self.CNN_channels, int(self.CNN_channels / 2), kernel_size=4, stride=2,
                                        padding=1)
        self.conv2 = nn.ConvTranspose2d(int(self.CNN_channels / 2), self.num_channels, kernel_size=4, stride=2,
                                        padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Unecessary??? Flatten the input tensor
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.unflatten(x)
        x = self.relu(self.conv1(x))  #  Output: B * 16 * 14 * 14
        x = self.upsample = F.interpolate(x, scale_factor=self.upscale, mode='bilinear',
                                          align_corners=False)  #  Output: B * 8 * 112 * 112
        x = self.sigmoid(self.conv2(x))  # Output: B * 3 * 224 * 224
        x = x.view(-1, self.num_channels, self.image_size, self.image_size)  #  shouldnt be necessary
        return x


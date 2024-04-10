import torch
import torch.nn as nn
import torchvision.models as models

#### BEGIN OLD ARCHITECTURE
# class MaskedAutoencoder(nn.Module):
#     """
#     Defines encoder / decoder for masked autoencoder pre-trainer
#     Takes
#         batch of images
#     Returns
#         image passed through encoder and decoder
#     """
#
#     def __init__(self, encoder, decoder):
#         super(MaskedAutoencoder, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# class VisionTransformerEncoder(nn.Module):
#     """
#     Defines vision transformer model
#     Takes
#         batch of images
#     Returns
#         vision-transformer feature embeddings of those images
#     """
#
#     def __init__(self, params):
#         super(VisionTransformerEncoder, self).__init__()
#         self.image_size = params['image_size']
#         self.patch_size = params['patch_size']
#         self.num_features = params['vit_num_features']
#         self.num_layers = params['vit_num_layers']
#         self.num_heads = params['vit_num_heads']
#         self.hidden_dim = params['vit_hidden_dim']
#         self.mlp_dim = params['vit_mlp_dim']
#         self.vit = models.vision_transformer.VisionTransformer(image_size=self.image_size,
#                                                                patch_size=self.patch_size,
#                                                                num_classes=self.num_features,
#                                                                num_layers=self.num_layers,
#                                                                num_heads=self.num_heads,
#                                                                hidden_dim=self.hidden_dim,
#                                                                mlp_dim=self.mlp_dim)
#
#     def forward(self, x):
#         # Pass the input through the ViT-B_16 backbone
#         features = self.vit(x)
#         return features
#
#
# # Define color image decoder
# class VisionTransformerDecoder(nn.Module):
#     """
#     Decoder
#     Takes
#        batch of image feature embeddings
#     Returns
#         reconstructed image tensor: Batch * Channels (3) * Image size (224) * Image size (224)
#
#     In the mvae process the three output dimensions are the three colours,
#     in the segmentation process these get re-interpretted as probabilities
#     over the three classes (foreground, background, boundary) - bit of a hack
#     """
#
#     def __init__(self, params):
#         super(VisionTransformerDecoder, self).__init__()
#         self.input_dim = params['vit_num_features']
#         self.image_size = params['image_size']
#         self.num_channels = params['num_channels']
#         self.output_dim = self.image_size * self.image_size * self.num_channels
#         self.hidden_dim_one = params['decoder_hidden_dim']
#         self.CNN_channels = params['decoder_CNN_channels']
#         self.upscale = params['decoder_scale_factor']
#         self.CNN_patch = int(self.image_size / 2 ** 2 / self.upscale)
#
#         self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_one)
#         self.fc2 = nn.Linear(self.hidden_dim_one, self.CNN_channels * self.CNN_patch * self.CNN_patch)
#         self.unflatten = nn.Unflatten(1, (self.CNN_channels, self.CNN_patch, self.CNN_patch))
#         self.conv1 = nn.ConvTranspose2d(self.CNN_channels, int(self.CNN_channels / 2), kernel_size=4, stride=2,
#                                         padding=1)
#         self.conv2 = nn.ConvTranspose2d(int(self.CNN_channels / 2), self.num_channels, kernel_size=4, stride=2,
#                                         padding=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
#
#         ## NEW- include normalisation
#         self.batch_norm_fc1 = nn.BatchNorm1d(self.hidden_dim_one)
#         self.batch_norm_fc2 = nn.BatchNorm1d(self.CNN_channels * self.CNN_patch * self.CNN_patch)
#         self.batch_norm_conv1 = nn.BatchNorm2d(int(self.CNN_channels / 2))
#         self.batch_norm_conv2 = nn.BatchNorm2d(self.num_channels)
#
#     ## NEW FORWARD
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.batch_norm_fc1(x)
#         x = self.relu(self.fc2(x))
#         x = self.batch_norm_fc2(x)
#         x = self.unflatten(x)
#         x = self.relu(self.conv1(x))
#         x = self.batch_norm_conv1(x)
#         x = F.interpolate(x, scale_factor=self.upscale,
#                           mode='bilinear',
#                           align_corners=False)
#         x = self.relu(self.conv2(x))
#         x = self.batch_norm_conv2(x)
#         x = self.sigmoid(x)
#         return x.view(-1, self.num_channels, self.image_size, self.image_size)
#### END OLD ARCHITECTURE


class CNNEncoder(nn.Module):
    def __init__(self, params):
        super(CNNEncoder, self).__init__()
        self.num_features = params['num_features']
        self.num_channels = params['num_channels']
        self.hidden_dim = params['hidden_dim']
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=8, kernel_size=3, stride=1, padding=1) #in 224, out 112
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1) #in 112 out 56
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) #in 56 out 28
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) #in 28 out 14
        self.fc1 = nn.Linear(64*14*14, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.num_features)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 64*14*14)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return x #output batch * num_features output


class CNNDecoder(nn.Module):
    """
        I've made output channels an explicit parameter to enable the same code to be used both for pixel generation
        (in which case the outputs are the 3 primary colours) and for pixel classification (in which case the outputs
        are the number of classes, which in the pet dataset happens to be 3, but this is coincidental)
    """
    def __init__(self, params, output_channels):
        super(CNNDecoder, self).__init__()
        self.num_features = params['num_features']
        self.output_channels = output_channels
        self.fc = nn.Linear(self.num_features, 512 * 14 * 14)
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(64, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.bnc1 = nn.BatchNorm2d(256)
        self.bnc2 = nn.BatchNorm2d(128)
        self.bnc3 = nn.BatchNorm2d(64)
        self.bnc4 = nn.BatchNorm2d(self.output_channels)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 14, 14)
        x = self.bnc1(self.relu(self.conv1(x))) # output 28
        x = self.bnc2(self.relu(self.conv2(x))) # output 56
        x = self.bnc3(self.relu(self.conv3(x))) # output 112
        x = torch.sigmoid(self.bnc4(self.conv4(x))) #output 224 * 224
        #x = torch.softmax(x, dim=1) # normalise to [0,1]

        return x # output batch * output_channels * image_size * image_size


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
        self.num_features = params['num_features']
        self.num_layers = params['vit_num_layers']
        self.num_heads = params['vit_num_heads']
        self.hidden_dim = params['hidden_dim']
        self.mlp_dim = params['vit_mlp_dim']
        self.vit = models.vision_transformer.VisionTransformer(image_size = self.image_size,
                                                               patch_size = self.patch_size,
                                                               num_classes = self.num_features,
                                                               num_layers = self.num_layers,
                                                               num_heads = self.num_heads,
                                                               hidden_dim = self.hidden_dim,
                                                               mlp_dim = self.mlp_dim,
                                                               dropout=0.1)

    def forward(self, x):
        # Pass the input through the ViT-B_16 backbone
        features = self.vit(x)
        return features


class LinearEncoder(nn.Module):
    def __init__(self, params):
        super(LinearEncoder, self).__init__()
        self.output_dim = params['num_features']
        self.hidden_dim = params['hidden_dim']
        self.image_size = params['image_size']
        self.num_channels = params['num_channels']
        self.fc1= nn.Linear(self.num_channels*self.image_size*self.image_size, self.hidden_dim)
        self.fc2= nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        bs = x.size(0)
        x = x.view(bs, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class SegmentModel(nn.Module):
    """
    Takes
        batch of images
    Returns
        either:
         - class probability per channel-pixel
         - value of each color pixel
    """
    def __init__(self, encoder, decoder):
        super(SegmentModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x) # image to features
        decoded = self.decoder(encoded) # features to segmentation map
        return decoded


# Function to get networks
def get_network(params, output_channels):
    match params['network']:
        case "CNN":
            return CNNEncoder(params), CNNDecoder(params, output_channels)
        case "ViT":
            return VisionTransformerEncoder(params), CNNDecoder(params, output_channels)
        case "Linear":
            return LinearEncoder(params), CNNDecoder(params, output_channels)
        case _:
            print("Network not known")

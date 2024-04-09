from torch import nn
import torch


class SemanticSegmenter(nn.Module):
    """
    Takes
        batch of images
    Returns
        class probability per channel-pixel
    """

    def __init__(self, encoder, decoder):
        super(SemanticSegmenter, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        classified = self.decoder(encoded)
        return classified


class SegmentationClassifier(nn.Module):
    """
    Takes
        image
    Returns
        class probability per channel-pixel
    """

    def __init__(self, params, num_classes):
        self.image_size = params['image_size']
        self.in_channels = params['num_channels']
        self.out_channels = params['CNN_channels']
        self.kernel_size = params['CNN_kernel']
        super(SegmentationClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels,
                               kernel_size=self.kernel_size, padding=1)
        self.upsample = nn.Upsample(size=(self.image_size, self.image_size), mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(in_channels=self.out_channels, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)

        return x



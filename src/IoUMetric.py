import torch
from torch import nn


class IoULoss(nn.Module):
    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax

    def forward(self, preds, ground_truth):
        """
        Compute IoU loss.
        Negative log loss is more stable that alternative (which is commented out).
        :param preds: Model's per-pixel class predictions. Tensor with shape (batch_size,3 , height, width).
        :param ground_truth: The ground-truth masks that indicate the correct class for each pixel in the image.
        Tensor with shape (batch_size, 1, height, width). The 2nd dimension indicates that for each image in batch,
        there is a single matrix of labels, with each entry in matrix corresponding to class label of a pixel.
        :return: Intersection over Union loss.
        """
        # return 1.0 - IoUMetric(pred, grount_truth, self.softmax)
        return -(iou_metric(preds, ground_truth, self.softmax).log())  # neg log loss for more stable training


def iou_metric(preds, ground_truth, preds_are_logits=False):
    """
    Calculate ratio of intersection over union for each image in batch averaged across the batch, known as intersection
    over union (IoU) metric to evaluate the model. It uses probabilities of per-pixel predictions.
    Intersection is an element-wise multiplication of the one-hot encoded ground-truth and predictions tensors,
    measuring true positives.
    Union is the sum of one-hot encoded ground-truth and predictions tensors minus their intersection, measuring the
    total area covered by both.
    :param preds: Model's per-pixel class predictions. Tensor with shape (batch_size, 3, height, width).
    :param ground_truth: The ground-truth masks that indicate the correct class for each pixel in the image. Tensor
    with shape (batch_size, 1, height, width). The 2nd dimension indicates that for each image in batch,
    there is a single matrix of labels, with each entry in matrix corresponding to class label of a pixel.
    :param preds_are_logits: True if `preds` input is expected to be logits.
    :return: IoU metric for batch.
    """
    if preds_are_logits is True: preds = nn.Softmax(dim=1)(preds)
    ones = []
    for i in range(3):
        ones.append(ground_truth == i)
    ground_truth = torch.cat(ones, dim=1)  # make one-hot encoded mask across all 3 classes.

    intersection = ground_truth * preds
    union = ground_truth + preds - intersection
    # Compute sum over all dimensions except batch_size dimension 0. Avoid divide by zero.
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.0001) / (union.sum(dim=(1, 2, 3)) + 0.0001)
    iou_mean = iou.mean()
    return iou_mean

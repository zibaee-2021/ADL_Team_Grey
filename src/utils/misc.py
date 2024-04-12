# GROUP19_COMP0197

def normalise(image_tensor):
    min_vals = image_tensor.view(image_tensor.size(0), -1).min(1, keepdim=True)[0].view(-1, 1, 1, 1)
    max_vals = image_tensor.view(image_tensor.size(0), -1).max(1, keepdim=True)[0].view(-1, 1, 1, 1)
    image_tensor = (image_tensor - min_vals) / (max_vals - min_vals)
    return image_tensor

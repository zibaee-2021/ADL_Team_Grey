import wandb
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from PIL import Image
import requests

dataset = load_dataset("richwardle/reduced-imagenet")

wandb.init(project="wandb_demo")

sweep_config = {
    'method': 'random'
    }


processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
loss = outputs.loss
mask = outputs.mask
ids_restore = outputs.ids_restore
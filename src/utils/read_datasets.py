import os
from time import time
from datasets import load_dataset
from PIL import Image

if __name__ == '__main__':
    # LOAD DATASET:
    start = time()
    # HF_dataset = "dgrnd4/animals-10"
    HF_dataset = "richwardle/reduced-imagenet"
    dataset = load_dataset(HF_dataset)
    print(f'Finished in {round(time() - start)} secs')

    # GO THROUGH DATASET AND GET SIZES AND CHANNELS:
    start = time()
    unique_dimensions = set()
    unique_modes = set()
    RGBA, L, RGB, CMYK = 0, 0, 0, 0
    for item in dataset['train']:
        img = item['image']
        unique_dimensions.add(img.size)
        unique_modes.add(img.mode)
        if img.mode == 'RGBA': RGBA += 1
        if img.mode == 'L': L += 1
        if img.mode == 'RGB': RGB += 1
        if img.mode == 'CMYK': CMYK += 1
        print(f"Image Size: {img.size}, Mode: {img.mode}")
    print(f'Unique dimensions found: {unique_dimensions}')
    print(f'Unique modes found: {unique_modes}')

    print(f'RGBA count = {RGBA}')
    print(f'L count = {L}')
    print(f'RGB count = {RGB}')
    print(f'CMYK count = {CMYK}')
    print(f'Finished in {round(time() - start)} secs')

    # CONVERT ALL TO RGB:
    start = time()
    desired_format = 'RGB'
    converted_images = []
    for item in dataset['train']:
        img = item['image']
        converted_img = img.convert(desired_format)
        converted_images.append(converted_img)
        # For demonstration, print size & mode of first few converted images
        if len(converted_images) <= 5:  # print first 5
            print(f"Converted Image Size: {converted_img.size}, Mode: {converted_img.mode}")
    print(f'Finished in {round(time() - start)} secs')

    # RESIZE ALL IMAGES TO SAME (224,224):
    start = time()
    desired_size = (224, 224)
    resized_images = []
    for img in converted_images:  # Assuming 'converted_images' is a list of your images
        print(f"Before resizing, size = {img.size}")
        resized_img = img.resize(desired_size, Image.Resampling.LANCZOS)
        print(f"After resizing, size = {resized_img.size}")
        resized_images.append(resized_img)
    print(f'Finished in {round(time() - start)} secs')

    # OR RESIZE, MAINTAINING ASPECT RATIO:
    def resize_maintaining_aspect_ratio():
        start = time()

        def resize_and_pad(img, desired_size):
            # Calculate the ratio of the height and perform the initial resizing

            ratio = max(desired_size[0] / img.size[0], desired_size[1] / img.size[1])
            intermediate_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(intermediate_size, Image.Resampling.LANCZOS)

            # Create new image with desired size and black background
            new_img = Image.new('RGB', desired_size)

            # Compute the positioning of the new image
            x = (desired_size[0] - intermediate_size[0]) // 2
            y = (desired_size[1] - intermediate_size[1]) // 2

            # Paste the resized image onto the new background
            new_img.paste(img, (x, y))
            return new_img
        desired_size = (224, 224)
        resized_images = []

        for img in converted_images:
            print(f"Before resizing, size = {img.size}")
            resized_img = resize_and_pad(img, desired_size)
            print(f"After resizing, size = {resized_img.size}")
            resized_images.append(resized_img)

        print(f'Finished in {round(time() - start)} secs')


    # AGAIN, GO THROUGH DATASET, GET SIZES & CHANNELS TO CHECK THEY'RE ALL RGB & (224, 224):
    start = time()
    unique_dimensions = set()
    unique_modes = set()
    RGBA, L, RGB, CMYK = 0, 0, 0, 0
    for img in resized_images:
        unique_dimensions.add(img.size)
        unique_modes.add(img.mode)
        if img.mode == 'RGBA': RGBA += 1
        if img.mode == 'L': L += 1
        if img.mode == 'RGB': RGB += 1
        if img.mode == 'CMYK': CMYK += 1
        print(f"Image Size: {img.size}, Mode: {img.mode}")
    # Print out unique dimensions and modes found
    print(f'Unique dimensions found: {unique_dimensions}')
    print(f'Unique modes found: {unique_modes}')
    print(f'RGBA count = {RGBA}')
    print(f'L count = {L}')
    print(f'RGB count = {RGB}')
    print(f'CMYK count = {CMYK}')
    print(f'Finished in {round(time() - start)} secs')

    # WRITE OUT ALL DATASET TO PNG FILES:
    start = time()
    # If directory not exist, make it:
    output_dir = '../src/datasets/Reduced_ImageNet'
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(resized_images):
        file_path = os.path.join(output_dir, f'image_{i}.png')
        img.save(file_path, 'PNG')

    print(f'Finished in {round(time() - start)} secs')

    # Note, we might need access token for other HF libraries, e.g.
    # access_token = "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"

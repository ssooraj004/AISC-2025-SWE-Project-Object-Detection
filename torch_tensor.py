import os
import torch
import json

from PIL import Image
import torchvision.transforms as transforms

# the root directory containing multiple subfolders of images
root_dir = 'furniture' 
save_dir = 'json_imgs'  # new directory to store the tensors

transform = transforms.Compose([
    transforms.PILToTensor()
])

# go through the folders and save tensors
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(subdir, file)
            try:
                image = Image.open(image_path).convert("RGB")
                img_tensor = transform(image)

                # convert tensor to list
                tensor_list = img_tensor.tolist()

                # consistent file name based on relative path
                relative_path = os.path.relpath(image_path, root_dir)
                json_filename = os.path.splitext(relative_path)[0] + '.json'
                json_save_path = os.path.join(save_dir, json_filename)

                # make any necessary directories
                os.makedirs(os.path.dirname(json_save_path), exist_ok=True)

                # save the tensor as JSON
                with open(json_save_path, 'w') as f:
                    json.dump(tensor_list, f)
                print(f"Saved JSON: {json_save_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
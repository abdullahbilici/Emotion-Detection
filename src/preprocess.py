from PIL import Image
import os

image_directory = "data/angry"
processed_directory = "data/processed"

for file_name in os.listdir(image_directory):
    if file_name.endswith(".jpg") or file_name.endswith(".png"): 
        image_path = os.path.join(image_directory, file_name)
        with Image.open(image_path) as img:
            gray_img = img.convert("L")
            resized_img = gray_img.resize((64, 64))
            processed_path = os.path.join(processed_directory, file_name)
            resized_img.save(processed_path)

print("Image processing complete!")
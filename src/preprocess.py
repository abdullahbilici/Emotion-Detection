from PIL import Image
import os

emotions = ["sad", "angry", "shocked", "happy"]

image_directory = "data/images"
processed_directory = "data/processed"

for emotion in emotions:
    emo_img_dir = os.path.join(image_directory, emotion)
    emo_pro_dir = os.path.join(processed_directory, emotion)

    for file_name in os.listdir(emo_img_dir):

        if file_name.endswith(".jpg") or file_name.endswith(".png"): 
            image_path = os.path.join(emo_img_dir, file_name)
            with Image.open(image_path) as img:
                gray_img = img.convert("L")
                resized_img = gray_img.resize((128, 128))
                processed_path = os.path.join(emo_pro_dir, file_name)
                resized_img.save(processed_path)

print("Image processing complete!")
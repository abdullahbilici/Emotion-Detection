from PIL import Image
import numpy as np
import os

def preprocess():
    emotions = ["angry", "happy", "sad", "shocked"]
    
    X = list()
    image_directory = "data/images"

    for i, emotion in enumerate(emotions):
        
        emotion_path = os.path.join(image_directory, emotion)

        for file_name in os.listdir(emotion_path):

            if file_name.endswith(".jpg") or file_name.endswith(".png"): 
                image_path = os.path.join(emotion_path, file_name)
                with Image.open(image_path) as img:
                    gray_img = img.convert("L")
                    resized_img = gray_img.resize((128, 128))
                    resized_img = np.array(resized_img).flatten() / 255
                    resized_img -= resized_img.mean()

                    resized_img = np.append(resized_img, i)
                    X.append(resized_img)

    X = np.array(X)

    data_path = "data/data"
    np.save(data_path, X)



if __name__ == "__main__":
    preprocess()


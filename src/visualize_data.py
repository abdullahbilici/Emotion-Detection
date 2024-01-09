import matplotlib.pyplot as plt
import random
import numpy as np

def visualize(data_path, grid_size=5, dimension = 128, ran = 1):
    
    data = np.load(data_path)
    dimensions = (dimension, dimension)

    emotion_dict = {0:"Angry", 1:"Happy", 2:"Sad", 3:"Shocked"}
    
    if ran == 1:
        sampled_data = random.sample(list(data), grid_size ** 2)
    else:
        sampled_data = data[:grid_size ** 2]
    
    plt.figure(figsize=(12, 12))
    for i, image_data in enumerate(sampled_data):
        image = image_data[:-1]
        label =image_data[-1]  
        image = image.reshape(dimensions)
        
        plt.subplot(grid_size, grid_size, i + 1) 
        plt.imshow(image, cmap = "gray")  
        plt.title(emotion_dict[label])  
        plt.axis("off")  

    plt.tight_layout()
    plt.show()

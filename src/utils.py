import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random

class DataLoader:
    def __init__(self, data, shuffle = False, batch_size = 1, device = "cpu", shape = None, transform=None):

        self.batch_size = batch_size
        self.shuffle = shuffle

        if isinstance(data, str):
            data = np.load(data)

        elif len(data) == 2:
            data = np.concatenate([data[0], data[1].reshape(-1,1)], axis = 1)



        self.X = torch.tensor(data[:,:-1]).to(device).view(-1, 1, *shape)
        self.y = torch.tensor(data[:,-1]).to(device)

        self.shape = tuple(self.X.shape)
        self.size = self.shape[0]
        
        self.transform = transform

    def __iter__(self):
        if self.shuffle:
            # Shuffle the data if True
            self._shuffle_data()

        # Create an iterator for batches
        self.current_index = 0
        return self

    def __next__(self):
        # Check if we have reached the end of the data
        if self.current_index >= self.size:
            raise StopIteration

        # Get the batch
        batch_x, batch_y = self.X[self.current_index : self.current_index + self.batch_size], self.y[self.current_index : self.current_index + self.batch_size]


        if self.transform:
            transformed_batch_x = []
            for x in batch_x:
                # Reshape and convert x to uint8
                x_reshaped = x.cpu().numpy().reshape(*self.shape[2:])
                x_reshaped = (x_reshaped * 255).astype(np.uint8)  # Normalize and convert to uint8
                transformed_x = self.transform(x_reshaped)
                transformed_batch_x.append(transformed_x)

            batch_x = torch.stack(transformed_batch_x)

        # Move the index to the next batch
        self.current_index += self.batch_size

        # Check if batch sizeis 1 or not
        if self.batch_size == 1:
            batch_x, batch_y = batch_x.squeeze(0), batch_y.item()
            
        return batch_x, batch_y

    def _shuffle_data(self):
        # Shuffle the data
        order = np.random.permutation(self.size)
        self.X = self.X[order]
        self.y = self.y[order]

    def __getitem__(self, inx: int):
        # If index is an integer
        if isinstance(inx, int):
            if inx < self.size:
                return self.X[inx], self.y[inx]
            else:
                raise IndexError
            
        # If index is a slice
        if isinstance(inx, slice):
            if inx.start < self.size:
                return self.X[inx], self.y[inx]
            else:
                raise IndexError
            
    def __len__(self):
        return self.size
        
    def __repr__(self):
        return f"Data with shape of {self.shape}, shuffle = {self.shuffle}, batch_size = {self.batch_size}"

def colorize_accuracy(test_acc):
    # Interpolate between red and green based on accuracy
    r = int((1 - test_acc) * 255)
    g = int(test_acc * 255)
    b = 0  # No blue component

    # Format the RGB values for ANSI escape code
    color_code = f"\033[38;2;{r};{g};{b}m"

    return color_code
    

def test_model(model, data_loader, criterion):
    model.eval()
    predictions = model(data_loader.X.float())
    pred_class = torch.argmax(predictions, axis = 1)


    emotions = ["Angry", "Happy", "Sad", "Shocked"]

    cm = confusion_matrix(data_loader.y.cpu(), pred_class.cpu())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=emotions,
                yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    
    acc = (pred_class == data_loader.y).float().mean()
    loss = criterion(predictions, data_loader.y.long()).item() / data_loader.size

    print(f"Loss: {loss :.4f}, Accuracy = {acc:.4f}")



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

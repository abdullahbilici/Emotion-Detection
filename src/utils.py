import torch
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class DataLoader:
    def __init__(self, data, shuffle = False, batch_size = 1, device = "cpu", shape = None, transforms=None):
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle

        if isinstance(data, str):
            data = np.load(data)

        elif len(data) == 2:
            data = np.concatenate([data[0], data[1].reshape(-1,1)], axis = 1)

        self.X = torch.tensor(data[:,:-1]).view(-1, 1, *shape)
        self.y = torch.tensor(data[:,-1])

        

        self.transforms = transforms
        self.augmented_data = self.X
        self.augmented_y = self.y
        if self.transforms:
            self.augmented_data = torch.stack([self.transforms[0](image).squeeze() for image in self.X]).view(-1, 1, *shape)
            for transform_ in self.transforms[1:]:
                augmented_data = torch.stack([transform_(image).squeeze() for image in self.X]).view(-1, 1, *shape)
                self.augmented_data = torch.cat([self.augmented_data, augmented_data], dim = 0)
                self.augmented_y = torch.cat([self.augmented_y, self.y], dim = 0)

        self.shape = tuple(self.augmented_data.shape)
        self.size = self.shape[0]
        
    def _shuffle_data(self):
        # Shuffle the data
        order = np.random.permutation(self.size)
        self.augmented_data = self.augmented_data[order]
        self.augmented_y = self.augmented_y[order]


    def __iter__(self):
        if self.transforms:
            self.augmented_data = torch.stack([self.transforms[0](image).squeeze() for image in self.X]).view(-1, 1, *self.shape[2:])
            self.augmented_y = self.y
            for transform in self.transforms[1:]:
                augmented_data = torch.stack([transform(image).squeeze() for image in self.X]).view(-1, 1, *self.shape[2:])
                self.augmented_data = torch.cat([self.augmented_data, augmented_data], dim = 0)
                self.augmented_y = torch.cat([self.augmented_y, self.y], dim = 0)

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
        batch_x, batch_y = self.augmented_data[self.current_index : self.current_index + self.batch_size], self.augmented_y[self.current_index : self.current_index + self.batch_size]

        # Move the index to the next batch
        self.current_index += self.batch_size

        # Check if batch size is 1 or not
        if self.batch_size == 1:
            batch_x, batch_y = batch_x.squeeze(0), batch_y.item()
            
        return batch_x.to(self.device), batch_y.to(self.device)

    def __getitem__(self, inx: int):
        # If index is an integer
        if isinstance(inx, int):
            if inx < self.size:
                return self.augmented_data[inx].to(self.device), self.augmented_y[inx].to(self.device)
            else:
                raise IndexError
            
        # If index is a slice
        if isinstance(inx, slice):
            if inx.start < self.size:
                return self.augmented_data[inx].to(self.device), self.augmented_y[inx].to(self.device)
            else:
                raise IndexError
            
        try:
            iter(inx)
            if np.all(np.array(inx) < self.size):
                return self.augmented_data[inx].to(self.device), self.augmented_y[inx].to(self.device)
            else:
                raise IndexError
        except:
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
    

def test_model(model, data_loader, criterion, device):
    model.eval()
    predictions = model(data_loader.X.to(device).float())
    pred_class = torch.argmax(predictions, axis = 1)


    emotions = ["Angry", "Happy", "Sad", "Shocked"]

    cm = confusion_matrix(data_loader.y, pred_class.cpu())

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=emotions,
                yticklabels=emotions)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    
    acc = (pred_class.cpu() == data_loader.y).float().mean()
    loss = criterion(predictions, data_loader.y.to(device).long()).item() / data_loader.size

    print(f"Loss: {loss :.4f}, Accuracy = {acc:.4f}")

def visualize(dataset, grid_size=5, dimension = 128):
    
    dimensions = (dimension, dimension)

    emotion_dict = {0:"Angry", 1:"Happy", 2:"Sad", 3:"Shocked"}
    indexes = np.random.choice(dataset.size, 25,replace=False)
    plt.figure(figsize=(12, 12))
    for i, (image_data, label) in enumerate(zip(dataset[indexes][0], dataset[indexes][1])):
        image = image_data.view(dimensions).cpu()
        
        plt.subplot(grid_size, grid_size, i + 1) 
        plt.imshow(image, cmap = "gray")  
        plt.title(emotion_dict[label.item()])  
        plt.axis("off")  

    plt.tight_layout()
    plt.show()

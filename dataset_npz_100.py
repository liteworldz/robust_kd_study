import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms 

class NumpyToTorchDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load the dataset
dataset = np.load('./cifar100_ddpm.npz')
images = dataset['image']
labels = dataset['label'].astype(np.int64)  # Convert labels to int64



def get_loader(val_size=50000, batch_size=2000):
    val_size=10000
    batch_size=800
    
    # Divide original data into two equal parts 500k each
    #base_images, ignored_images, base_labels, ignored_labels = train_test_split(images, labels, test_size=0.8, random_state=42)
    # Divide the dataset into train, validation, and test sets
    train_val_images, test_images, train_val_labels, test_labels = train_test_split(images, labels, test_size=0.01, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_val_images, train_val_labels, test_size=val_size, random_state=42)

    # Define transforms for the images
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


    # Create PyTorch DataLoader objects for the train, validation, and test sets
    train_dataset = NumpyToTorchDataset(train_images, train_labels, transform=transform_train)
    

    val_dataset = NumpyToTorchDataset(val_images, val_labels, transform=transform_train)
    

    test_dataset = NumpyToTorchDataset(test_images, test_labels, transform=transform_test)
    
    
    return train_dataset, val_dataset, test_dataset, len(train_images)



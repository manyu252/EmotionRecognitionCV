import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataset():
    df = pd.read_csv('data/fer2013/fer2013/fer2013.csv')

    img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48).astype('float32'))
    img_array = np.stack(img_array, axis = 0)

    label_array = df.emotion.values

    print("data loaded")

    return img_array, label_array

def create_dataloader(img_array, label_array, batch_size, train_ratio, augment):
    print("Batch size: ", batch_size)

    # Define the augmentation transformations
    if augment:
        image_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust color jitter
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image pixels
        ])
    else:
        image_transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image pixels
        ])

    # Apply the augmentation transformations to the dataset
    pil_images = [Image.fromarray(image).convert('RGB') for image in img_array]
    augmented_dataset = CustomDataset(pil_images, label_array, transform=image_transforms)

    # Create a DataLoader instance for the augmented dataset
    augmented_data_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)

    # Determine the size of the training and testing sets
    dataset_size = len(augmented_data_loader.dataset)
    train_size = int(train_ratio * dataset_size)  # 80% for training, adjust as needed
    test_size = dataset_size - train_size

    # Split the dataset into training and testing sets
    train_dataset, test_dataset = random_split(augmented_data_loader.dataset, [train_size, test_size])

    # Create separate DataLoaders for training and testing
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("dataloaders created")
    return train_data_loader, test_data_loader

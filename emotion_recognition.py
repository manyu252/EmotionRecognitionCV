import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim


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

class EmoCNN(nn.Module):
    def __init__(self):
        super(EmoCNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (1, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # (16, 48, 48)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 48, 48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (32, 24, 24)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (64, 12, 12)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 12 * 12, 7) # 7 classes

    def forward(self, x):
        x = self.layer1(x) # (16, 48, 48)
        x = self.layer2(x) # (32, 24, 24)
        x = self.layer3(x) # (64, 12, 12)
        x = self.layer4(x) # (128, 12, 12)
        x = self.layer5(x) # (128, 12, 12)
        x = x.reshape(x.size(0), -1) # (128 * 12 * 12)
        x = self.fc(x)
        return x

class Emo2CNN(nn.Module):
    def __init__(self):
        super(Emo2CNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (3, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0), # (16, 46, 46)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0), # (32, 44, 44)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (32, 22, 22)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), # (64, 20, 20)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (64, 10, 10)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 10, 10)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 10, 1)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 10 * 10, 7) # 7 classes

    def forward(self, x):
        x = self.layer1(x) # (16, 46, 46)
        x = self.layer2(x) # (32, 22, 22)
        x = self.layer3(x) # (64, 10, 10)
        x = self.layer4(x) # (128, 10, 10)
        x = self.layer5(x) # (128, 10, 10)
        x = x.reshape(x.size(0), -1) # (128 * 10 * 10)
        x = self.fc(x)
        return x

class Emo3CNN(nn.Module):
    def __init__(self):
        super(Emo3CNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (3, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # (16, 48, 48)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 48, 48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (32, 24, 24)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (64, 12, 12)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc = nn.Linear(256 * 12 * 12, 7) # 7 classes

    def forward(self, x):
        x = self.layer1(x) # (16, 48, 48)
        x = self.layer2(x) # (32, 24, 24)
        x = self.layer3(x) # (64, 12, 12)
        x = self.layer4(x) # (128, 12, 12)
        x = self.layer5(x) # (128, 12, 12)
        x = self.layer6(x) # (256, 12, 12)
        x = x.reshape(x.size(0), -1) # (256 * 12 * 12)
        x = self.fc(x)
        return x

class Emo4CNN(nn.Module):
    def __init__(self):
        super(Emo4CNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (3, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # (16, 48, 48)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 48, 48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (32, 24, 24)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (64, 12, 12)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(256 * 12 * 12, 128) # fc 256 -> 128
        self.fc2 = nn.Linear(128, 7) # 7 classes

    def forward(self, x):
        x = self.layer1(x) # (16, 48, 48)
        x = self.layer2(x) # (32, 24, 24)
        x = self.layer3(x) # (64, 12, 12)
        x = self.layer4(x) # (128, 12, 12)
        x = self.layer5(x) # (128, 12, 12)
        x = self.layer6(x) # (256, 12, 12)
        x = x.reshape(x.size(0), -1) # (256 * 12 * 12)
        x = self.fc1(x)
        x = x.reshape(x.size(0), -1) # (128 * 12 * 12)
        x = self.fc2(x)
        return x

class Emo5CNN(nn.Module):
    def __init__(self):
        super(Emo5CNN, self).__init__()
        # Add more layers as per your desired architecture
        self.layer1 = nn.Sequential( # input shape (3, 48, 48)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # (16, 48, 48)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 48, 48)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (32, 24, 24)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (64, 24, 24)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # (64, 12, 12)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # (256, 12, 12)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1), # (128, 12, 12)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), # (64, 12, 12)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(64 * 12 * 12, 7) # fc 64 * 12 * 12 -> 7

    def forward(self, x):
        x = self.layer1(x) # (16, 48, 48)
        x = self.layer2(x) # (32, 24, 24)
        x = self.layer3(x) # (64, 12, 12)
        x = self.layer4(x) # (128, 12, 12)
        x = self.layer5(x) # (128, 12, 12)
        x = self.layer6(x) # (256, 12, 12)
        x = self.layer7(x) # (256, 12, 12)
        x = self.layer8(x) # (128, 12, 12)
        x = self.layer9(x) # (64, 12, 12)
        x = x.reshape(x.size(0), -1) # (64 * 12 * 12)
        x = self.fc1(x)
        return x

def create_dataset():
    df = pd.read_csv('data/fer2013/fer2013/fer2013.csv')

    img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48).astype('float32'))
    img_array = np.stack(img_array, axis = 0)

    label_array = df.emotion.values

    print("data loaded")

    return img_array, label_array

def create_dataloader(img_array, label_array, batch_size, train_ratio):
    # image_transforms = transforms.Compose([
    #     transforms.ToTensor(),  # Convert the image to a tensor
    #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image pixels
    # ])

    # Assuming you have your images in an array called 'image_array' and labels in 'label_array'
    # pil_images = [Image.fromarray(image).convert('RGB') for image in img_array]
    # dataset = CustomDataset(pil_images, label_array, transform=image_transforms)

    # Create a DataLoader instance to load the dataset
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the augmentation transformations
    augmentation_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust color jitter
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image pixels
    ])

    # Apply the augmentation transformations to the dataset
    pil_images = [Image.fromarray(image).convert('RGB') for image in img_array]
    augmented_dataset = CustomDataset(pil_images, label_array, transform=augmentation_transforms)

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

def select_model(model_name):
    if model_name == "Emo5CNN":
        model = Emo5CNN()
    if model_name == "Emo4CNN":
        model = Emo4CNN()
    elif model_name == "Emo3CNN":
        model = Emo3CNN()
    elif model_name == "Emo2CNN":
        model = Emo2CNN()
    elif model_name == "EmoCNN":
        model = EmoCNN()
    print("model selected: {}".format(model_name))
    return model

def lr_update_bad(lr, epoch, loss, lowest_loss):
    if (epoch >= 5 and lr > 0.00001):
        if loss <= lowest_loss:
            lr = lr / 4
        elif loss * 10 > lowest_loss:
            lr = lr / 10
        print("New Learning rate: {}".format(lr))
    return lr

def lr_update(lr, epoch):
    # Adjust learning rate by dividing by 10 every 10 epochs
    if epoch == 10:
        lr = 0.0001
        print("New Learning rate: {}".format(lr))
    elif epoch == 20:
        lr = 0.00001
        print("New Learning rate: {}".format(lr))
    return lr

def train(train_data_loader, device, n_epochs, model_name, lr):
    # Create an instance of the model class
    model = select_model(model_name)
    # Move the model to the GPU
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_epoch = 0
    lowest_loss = np.Inf
    train_loss = []
    start_time = time.time()
    # Iterate over epochs
    for epoch in range(n_epochs):
        model.train()  # Set model to training mode

        # # Adjust the learning rate, dividing it by 4 whenever the loss plateaus after an epoch and divide it by 10 when the loss reduces by a factor of 10
        # if (epoch >= 5 and train_loss[-1] >= train_loss[-2]):
        #     lr = lr / 4
        #     print("Learning rate: {}".format(lr))
        # elif (epoch >= 5 and train_loss[-1] * 10 <= train_loss[-2]):
        #     lr = lr / 10
        #     print("Learning rate: {}".format(lr))

        # Adjust the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print("Epoch: {}/{}".format(epoch + 1, n_epochs), end="\t")
        # Iterate over the dataset
        for images, labels in train_data_loader:
            optimizer.zero_grad()  # Zero out gradients
            output = model(images.to(device))  # Forward pass
            loss = criterion(output, labels.to(device))  # Compute loss
            loss.backward()  # Perform backward pass
            optimizer.step()  # Update weights
        
        print("Loss: {:.6f}".format(loss.item()))
        train_loss.append(loss.item())
        if loss.item() < lowest_loss:
            # lr = lr_update(lr, epoch, loss, lowest_loss)
            lr = lr_update(lr, epoch)
            lowest_loss = loss.item()
            torch.save(model.state_dict(), 'model.pt')
            # print("Model saved: Epoch: {}, Loss: {}\n\n".format(epoch+1, lowest_loss))
            best_epoch = epoch + 1

    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")
    print(f"Best epoch: {best_epoch} with loss: {lowest_loss}")
    return train_loss

def test(test_data_loader, device, model_name):
    model = select_model(model_name)
    state_dict = torch.load('model.pt')
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_data_loader:
            labels = labels.to(device)
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy = {accuracy}%")

def arg_parser():
    # create argument parser for command line arguments like epochs, batch size, model type, learning rate, etc.
    parser = argparse.ArgumentParser(description='Train a CNN for emotion recognition')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--model', type=str, default='Emo3CNN', help='model to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='ratio of training data to total data')
    parser.add_argument('--device', type=str, default='mps', help='device to train on')
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    img_array, label_array = create_dataset()
    train_data_loader, test_data_loader = create_dataloader(img_array, label_array, args.batch_size, args.train_ratio)
    train(train_data_loader, args.device, args.epochs, args.model, args.lr)
    test(test_data_loader, args.device, args.model)

if __name__ == "__main__":
    main()
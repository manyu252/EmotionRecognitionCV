import argparse
import shutil
import os
import datetime
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from data_loader import create_dataset, create_dataloader
from models.EmoCNN import EmoCNN
from models.Emo2CNN import Emo2CNN
from models.Emo3CNN import Emo3CNN
from models.Emo4CNN import Emo4CNN
from models.Emo5CNN import Emo5CNN
from models.Emo6CNN import Emo6CNN

def labels_mapping(label):
    if label == 0:
        return "Angry"
    elif label == 1:
        return "Disgust"
    elif label == 2:
        return "Fear"
    elif label == 3:
        return "Happy"
    elif label == 4:
        return "Sad"
    elif label == 5:
        return "Surprise"
    elif label == 6:
        return "Neutral"


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
    elif model_name == "Emo6CNN":
        model = Emo6CNN()
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

def lr_update(lr, epoch, adaptive_lr, lr_decay_after_epochs):
    if not adaptive_lr:
        return lr

    # Adjust learning rate by dividing by 10 every few epochs
    if ((epoch+1) % lr_decay_after_epochs) == 0:
        lr = lr / 10
        print("New Learning rate: {}".format(lr))
    return lr

def train(train_data_loader, val_data_loader, device, n_epochs, model_name, lr, adaptive_lr, batch_size, lr_decay_after_epochs, output_dir):
    print("Device: {}".format(device))
    print("Training for {} epochs".format(n_epochs))
    print("Learning rate: {}".format(lr))

    # Create an instance of the model class
    model = select_model(model_name)
    # Move the model to the GPU
    model = model.to(device)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_epoch = 0
    lowest_loss = np.Inf
    train_losses = []
    eval_losses = []

    start_time = time.time()
    # Iterate over epochs
    for epoch in range(n_epochs):
        epoch_train_start = time.time()

        train_loss = 0
        model.train()  # Set model to training mode

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
            train_loss += loss.item()

        # Calculate average training loss for the epoch
        train_loss /= len(train_data_loader)
        train_losses.append(train_loss)

        lr = lr_update(lr, epoch, adaptive_lr, lr_decay_after_epochs)
        if train_loss < lowest_loss:
            lowest_loss = train_loss
            model_save_path = os.path.join(output_dir, 'model.pt')
            torch.save(model.state_dict(), model_save_path)
            best_epoch = epoch + 1

        epoch_train_end = time.time()
        print("Loss: {:.6f}".format(train_loss), end="\t")
        print("Epoch time: {:.2f}s".format(epoch_train_end - epoch_train_start))

        # Evaluate the model on the validation set
        epoch_eval_start = time.time()
        eval_loss = 0
        correct = 0
        total = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for inputs, labels in val_data_loader:
                labels = labels.to(device)
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate average evaluation loss for the epoch
            eval_loss /= len(val_data_loader)
            eval_losses.append(eval_loss)

            accuracy = 100 * correct / total
            epoch_eval_end = time.time()
            print("Validation Accuracy = {:.4f}%".format(accuracy), end="\t")
            print("Validation Time: {:.2f}s".format(epoch_eval_end - epoch_eval_start), end="\t")
            print("Inference Time: {:.2f}s\n".format((epoch_eval_end - epoch_train_start)/(len(val_data_loader)*batch_size)))

    end_time = time.time()
    print("Training time: {:.2f}s".format(end_time - start_time))
    print(f"Best epoch: {best_epoch} with loss: {lowest_loss}")

    # Print the model summary
    # summary(model, input_size=(3, 48, 48), batch_size=batch_size, device="cpu")

    # Plot the loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(eval_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot as a JPG file
    loss_plot_path = os.path.join(output_dir, 'loss_plot.jpg') 
    plt.savefig(loss_plot_path)

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

def read_config_json(file_path):
    try:
        with open(file_path) as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print("Config file not found")

def create_output_folder():
    try:
        output_dir = "outputs/output_" + str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        os.mkdir(output_dir)
    except FileExistsError:
        print("Output folder already exists")
    return output_dir

def copy_files_to_output_dir(output_dir, config_file_path):
    shutil.copyfile(config_file_path, os.path.join(output_dir, 'config.json'))

def argparser():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    return parser.parse_args()

def main():
    args = argparser()
    config = read_config_json(args.config)
    output_dir = create_output_folder()
    img_array, label_array = create_dataset()
    train_data_loader, val_data_loader = create_dataloader(img_array, label_array, config["batch_size"], config['train_ratio'], config['augment'])
    train(train_data_loader, val_data_loader, config['device'], config['num_epochs'], config['model'], config['lr'], config['adaptive_lr'], config['batch_size'], config['lr_decay_after_epochs'], output_dir)
    copy_files_to_output_dir(output_dir, args.config)
    # test(test_data_loader, args.device, args.model)

if __name__ == "__main__":
    main()
import math
import torch
import argparse
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms as transforms

from emotion_recognition import select_model

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the activations of a CNN')
    parser.add_argument('-i', '--image', type=str, help='Path to the image', required=True)
    parser.add_argument('-m', '--model', type=str, help='Model name', required=True)
    parser.add_argument('-w', '--model_path', type=str, help='Path to the model', required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load an image tensor
    image = Image.open(args.image).convert('RGB')

    # Define the transformation
    transform = transforms.ToTensor()

    # Apply the transformation to the image
    image_tensor = transform(image)

    # Load the CNN model
    model = select_model(args.model)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('mps')))

    # Get the activations from a specific layer
    image_tensor = image_tensor.unsqueeze(0)

    # Define a list to store the activations
    activations = []

    # Iterate over the layers
    x = image_tensor
    for name, layer in model.named_modules():
        # Check if the layer is a convolutional layer
        if isinstance(layer, torch.nn.Conv2d):
            print(name)
            # Get the activation from the layer
            activation = layer(x)
            x = activation
            activation = activation.detach().cpu().numpy()

            # Reshape and rearrange the dimensions
            activation = activation.squeeze()
            activation = activation.transpose(1, 2, 0)

            # Store the activation in the list
            activations.append((name, activation))


    # Create a subplot grid to display the activation maps
    num_columns = 3
    num_layers = len(activations)
    num_rows = math.ceil(num_layers / num_columns)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, 6))

    # Iterate over the activations and plot them in the subplot grid
    for idx, (name, activation) in enumerate(activations):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]
        ax.imshow(activation[...,0], cmap='jet')
        ax.axis('off')
        ax.set_title(f'Layer: {name}')

    fig.tight_layout()
    save_path = args.model_path.split('.pt')[0] + '_activations.jpg'
    plt.savefig(save_path)

if __name__ == '__main__':
    main()
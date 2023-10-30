import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(100, 256)  # 100-dimensional noise
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 30*30)  # Output a 30x30 image

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        environment = torch.tanh(self.fc3(x)).reshape(-1, 30, 30)  # Reshape to get a 30x30 image

        return environment

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(30*30, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)  # Binary classification: real or fake

    def forward(self, environment):
        # print("Input shape to discriminator:", environment.shape)
        x = environment.view(environment.size(0), -1)  # Flatten the image
        # print("Shape after flattening:", x.shape)
        x = torch.relu(self.fc1(x))
        # print("Shape after fc1:", x.shape)
        x = torch.relu(self.fc2(x))
        validity = self.fc3(x)

        return validity

import wandb

# Initialize wandb
wandb.init(project="maze-gan", name="gan_run_1")
# Define the checkpoint saving function
def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, path="checkpoint.pth.tar"):
    state = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }
    torch.save(state, path)
def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, path="checkpoint.pth.tar"):
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch
# def sample_images(generator, num_samples=100, save_path='./generated_images/'):
#     # Make sure the generator is in evaluation mode
#     generator.eval()
    
#     # Generate random noise for the generator
#     z = torch.randn((num_samples, 100))
#     with torch.no_grad():  # No need to compute gradients in the testing phase
#         sample_environments = generator(z).cpu().numpy()

#     # Reshape the generated images for visualization
#     sample_environments = sample_environments.reshape(num_samples, 30, 30)
    
#     # Save the images
#     for i, img in enumerate(sample_environments):
#         # Convert the generated environments to the range [0, 255]
#         img_normalized = ((img + 1) * 0.5 * 255).astype(np.uint8)
#         img_pil = Image.fromarray(img_normalized, 'L')  # 'L' mode indicates grayscale
#         img_pil.save(f"{save_path}generated_image_{i}.png")
    
#     return sample_environments
def sample_images(generator, num_samples=100, save_path='./generated_npy/'):
    # Make sure the generator is in evaluation mode
    generator.eval()
    
    # Generate random noise for the generator
    z = torch.randn((num_samples, 100))
    with torch.no_grad():  # No need to compute gradients in the testing phase
        sample_environments = generator(z).cpu().numpy()

    # Reshape the generated images for visualization
    sample_environments = sample_environments.reshape(num_samples, 30, 30)

    # Convert the images and save them
    for i, img in enumerate(sample_environments):
        # Convert to binary: 0 for white, 1 for black using a threshold of 0
        binary_img = np.where(img > 0, 0, 1)
        np.save(f"{save_path}generated_image_{i}.npy", binary_img)
    
    return sample_environments

# In this function:

# I added a save_path parameter to specify the directory where the images should be saved. Make sure this directory exists before running the function.
# I convert the image data from the range [-1, 1] (as output by the generator with the tanh activation) to the range [0, 255] for saving.
# I use the PIL library to save each image



# Training loop
def train(generator, discriminator, data_loader, g_optimizer, d_optimizer, checkpoint_interval, num_epochs=1000, ):


    # Watch the models
    wandb.watch(generator, log="all")
    wandb.watch(discriminator, log="all")

    for epoch in range(num_epochs):


        for real_environments in data_loader:
            real_environments = real_environments[0]  # Extracting images from the batch
            batch_size = real_environments.size(0)

            # Generate fake environments
            z = torch.randn((batch_size, 100))  # Random noise
            fake_environments = generator(z).detach()  # Detach the generator's outputs for discriminator training

            # Discriminator's turn
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            logits_real = discriminator(real_environments)
            logits_fake = discriminator(fake_environments)

            # Calculate the loss
            loss_real = nn.BCEWithLogitsLoss()(logits_real, real_labels)
            loss_fake = nn.BCEWithLogitsLoss()(logits_fake, fake_labels)
            d_loss = (loss_real + loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Generator's turn (using the attached outputs)
            fake_environments = generator(z)  # Don't detach here since we need to backpropagate through the generator
            logits_fake = discriminator(fake_environments)
            g_loss = nn.BCEWithLogitsLoss()(logits_fake, real_labels)  # Traditional GAN loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Log the losses
            wandb.log({"Generator Loss": g_loss.item(), "Discriminator Loss": d_loss.item()})

            # Optionally, log generated images for visualization
            if epoch % 10 == 0:  # Log every 10 epochs, adjust as needed
                fake_images = fake_environments.detach().cpu().numpy()
                # Reshape the images to (height, width) for visualization
                fake_images = fake_images.reshape(batch_size, 30, 30)
                wandb.log({"Generated Images": [wandb.Image(img, caption="Generated Image") for img in fake_images]})
        if epoch % checkpoint_interval == 0:
            save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, path=f"./gan_checkpoint/checkpoint_epoch_{epoch}.pth.tar")

# Load the dataset
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

dataset = datasets.ImageFolder(root='./image_folder/', transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
sample = next(iter(data_loader))
# print("Sample shape:", sample[0].shape)
checkpoint_interval = 100
num_epochs = 1000
# Initialize models and train
generator = Generator()
discriminator = Discriminator()
test = True
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
if not test:
    
    train(generator, discriminator, data_loader,g_optimizer, d_optimizer, checkpoint_interval, num_epochs = num_epochs)
else:
    load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, path=f"./gan_checkpoint/checkpoint_epoch_900.pth.tar")
    sample_images(generator, num_samples=100, save_path='./generated_npy/')
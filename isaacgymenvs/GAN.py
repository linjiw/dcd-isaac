import torch
import torch.nn as nn
import torch.optim as optim
import gym
# Assuming IsaacGym is appropriately installed and imported

# Define the Generator using PyTorch
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Assuming the input noise dimension is 100
        self.model = nn.Sequential(
            nn.Linear(100, 512*4*4),  # Fully connected layer to reshape noise
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # Upscale to 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # Upscale to 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # Upscale to 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # Upscale to 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # Upscale to 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),    # Upscale to 256x256
            nn.Tanh()  # Tanh activation to generate pixel values between -1 and 1
        )

    def forward(self, noise):
        x = self.model(noise)
        return x

# Define the Discriminator using PyTorch
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),  # Downscale to 128x128
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # Downscale to 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # Downscale to 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Downscale to 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # Downscale to 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # Downscale to 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Flatten(),
            nn.Linear(512*4*4, 1),
            nn.Sigmoid()  # Output a probability of image being real
        )

    def forward(self, map_image):
        validity = self.model(map_image)
        return validity


class Adversary:
    def __init__(self, generator, protagonist, antagonist):
        self.generator = generator
        self.protagonist = protagonist
        self.antagonist = antagonist

    def generate_map(self, batch_size=1):
        noise = torch.randn((batch_size, 100))
        return self.generator(noise)

    def calculate_regret(self, map_image):
        protagonist_score = self.protagonist.evaluate(map_image)
        antagonist_score = self.antagonist.evaluate(map_image)
        regret = antagonist_score - protagonist_score
        
        # The adversary's goal is to maximize this regret
        return regret

    def update(self, regret, optimizer):
        # Use the regret to update the generator
        # We aim to minimize the negative regret, as the goal is to increase it
        loss = -regret

        # Clear accumulated gradients
        optimizer.zero_grad()

        # Backpropagate the error
        loss.backward()

        # Update the generator's weights
        optimizer.step()

# Main loop for training the GAN with PAIRED
def train_gan(epochs, alpha=0.5, beta=0.5):
    generator = Generator()
    discriminator = Discriminator()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    protagonist = Agent("protagonist")
    antagonist = Agent("antagonist")
    adversary = Adversary(generator, protagonist, antagonist)

    for epoch in range(epochs):
        # Step 1: Train discriminator with real and fake images
        real_maps = get_real_maps() # This function fetches real map data
        fake_maps = adversary.generate_map(batch_size=len(real_maps))

        real_labels = torch.ones(len(real_maps))
        fake_labels = torch.zeros(len(fake_maps))

        preds_real = discriminator(real_maps)
        loss_real = F.binary_cross_entropy(preds_real, real_labels)
        
        preds_fake = discriminator(fake_maps.detach())
        loss_fake = F.binary_cross_entropy(preds_fake, fake_labels)
        
        d_loss = loss_real + loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Step 2: Train the generator using both GAN loss and regret
        preds_fake_g = discriminator(fake_maps)
        GAN_loss = F.binary_cross_entropy(preds_fake_g, real_labels)  
        
        regret = adversary.calculate_regret(fake_maps)
        
        # Weighted combination of GAN_loss and regret
        total_loss = alpha * GAN_loss + beta * regret
        optimizer_g.zero_grad()
        total_loss.backward()
        optimizer_g.step()

        # Optionally, train the protagonist and antagonist on the current generated map
        # This depends on the specific design of your agents and their training methods

        # Step 3: Provide feedback, logging and potentially save model weights.
        if epoch % 50 == 0:  # adjust the frequency as needed
            print(f"Epoch {epoch}/{epochs}, D Loss: {d_loss.item()}, G Loss: {GAN_loss.item()}, Regret: {regret.item()}")
            # Save model weights or other logging tasks


# Note:
# The code is now structured for PyTorch, but remember, integrating RL loop with the GAN training 
# loop can be complex. Ensure that the Agent's evaluate method is appropriately defined. 

train_gan(epochs=1000)

import numpy as np
from PIL import Image

def generate_occupancy_map(size=30):
    """Generate a random occupancy map."""
    return np.random.choice([255, 0], size=(size, size))  # 255 for white and 0 for black

def save_map_as_image(occupancy_map, filename):
    """Save the occupancy map as an image."""
    img = Image.fromarray(occupancy_map, 'L')  # 'L' mode is for 8-bit pixels, black and white
    img.save(filename)

# Generate and save 10 random occupancy maps
for i in range(500):
    map_ = generate_occupancy_map()
    save_map_as_image(map_, f"./maze/train/occupancy_map_{i}.png")

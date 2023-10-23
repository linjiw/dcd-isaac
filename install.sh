# #!/bin/bash

# # Exit on any error
# set -e

# # Check if conda is installed
# if ! command -v conda &> /dev/null
# then
#     echo "conda is not installed. Please install Miniconda or Anaconda and try again."
#     exit
# fi

# # Create a conda environment named 'myenv'
# conda create --name isaac-dcd python=3.8 -y

# # Activate the conda environment
# conda activate isaac-dcd

# # # Install required packages from environment.yml
# # if [ -f "environment.yml" ]; then
# #     conda env update --file environment.yml --prune
# # else
# #     echo "environment.yml not found. Skipping package installation."
# # fi
conda install matplotlib=3.3.4
pip install numpy==1.19
pip install wandb
# echo "Installation completed successfully!"

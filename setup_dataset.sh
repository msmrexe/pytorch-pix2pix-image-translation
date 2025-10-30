#!/bin/bash

echo "Setting up Kaggle API credentials..."

# Create the .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Check if kaggle.json exists in the root directory
if [ ! -f "kaggle.json" ]; then
    echo "Error: kaggle.json not found in the root directory."
    echo "Please download it from your Kaggle Account settings and place it here."
    exit 1
fi

# Move kaggle.json and set permissions
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle credentials configured."

echo "Downloading Cityscapes Pix2Pix dataset..."
kaggle datasets download -d balraj98/cityscapes-pix2pix-dataset

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Dataset download failed. Please check Kaggle credentials and network."
    exit 1
fi

echo "Unzipping dataset..."
unzip -q cityscapes-pix2pix-dataset.zip -d cityscapes_pix2pix_dataset

# Check if unzip was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip dataset."
    exit 1
fi

echo "Cleaning up zip file..."
rm cityscapes-pix2pix-dataset.zip

echo "Dataset setup complete. Data is in 'cityscapes_pix2pix_dataset/'"

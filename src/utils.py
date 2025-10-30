"""
Utility functions for logging, saving models, and visualizing results.
"""
import os
import logging
import torch
import matplotlib.pyplot as plt
from src import config

def setup_logging():
    """Configures the project logger."""
    os.makedirs("logs", exist_ok=True)
    
    # Basic configuration for file and console logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ])
    return logging.getLogger(__name__)

def save_checkpoint(model, optimizer, filename):
    """Saves model and optimizer state."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """Loads model and optimizer state."""
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer

def denormalize(tensor):
    """Denormalizes a tensor from [-1, 1] to [0, 1]."""
    return tensor * 0.5 + 0.5

def save_sample_images(model, loader, epoch, num_samples=3):
    """
    Saves a grid of input, target, and generated images during training.
    """
    model.eval()
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    
    with torch.no_grad():
        segmented_images, real_images = next(iter(loader))
        segmented_images = segmented_images[:num_samples].to(config.DEVICE)
        real_images = real_images[:num_samples].to(config.DEVICE)

        fake_images = model.generator(segmented_images)

        # Denormalize for plotting
        segmented_images = denormalize(segmented_images.cpu())
        real_images = denormalize(real_images.cpu())
        fake_images = denormalize(fake_images.cpu())

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        fig.suptitle(f"Generated Samples - Epoch {epoch}", fontsize=16, y=1.02)

        for i in range(num_samples):
            axes[i, 0].imshow(segmented_images[i].permute(1, 2, 0))
            axes[i, 0].set_title("Input (Segmented)")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(real_images[i].permute(1, 2, 0))
            axes[i, 1].set_title("Target (Real)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(fake_images[i].permute(1, 2, 0))
            axes[i, 2].set_title("Generated")
            axes[i, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(config.SAMPLE_DIR, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path)
        plt.close(fig)
    
    model.train()

def plot_and_save_losses(g_losses, d_losses, title, filename):
    """Plots and saves generator and discriminator loss curves."""
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.suptitle(title, fontsize=16)
    
    plt.plot(range(1, len(g_losses) + 1), g_losses, label="Generator Loss")
    plt.plot(range(1, len(d_losses) + 1), d_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Generator and Discriminator Loss Over Epochs")
    plt.legend()
    
    save_path = os.path.join(config.PLOT_DIR, filename)
    plt.savefig(save_path)
    plt.close()

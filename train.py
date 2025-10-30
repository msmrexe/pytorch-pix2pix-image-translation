"""
Main training script for the Pix2Pix model.
Handles argument parsing, model setup, training loop, and validation.
"""
import argparse
import time
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import warnings

# Suppress persistent_workers warning in DataLoader
warnings.filterwarnings("ignore", ".*UserWarning: 'persistent_workers' is experimental.*")

# Project modules
from src import config
from src.dataset import get_loaders
from src.models import Generator, Discriminator
from src.pix2pix_model import Pix2PixModel
from src.utils import (
    setup_logging,
    save_checkpoint,
    save_sample_images,
    plot_and_save_losses
)

# Setup logger
logger = setup_logging()

def train_one_epoch(model, loader, opt_g, opt_d):
    """Performs a single training epoch."""
    model.train()
    g_loss_total = 0.0
    d_loss_total = 0.0
    
    loop = tqdm(loader, desc="Training", leave=False)
    for segmented_images, real_images in loop:
        segmented_images = segmented_images.to(config.DEVICE)
        real_images = real_images.to(config.DEVICE)
        
        # 1. Train Discriminator
        opt_d.zero_grad()
        fake_images = model.generator(segmented_images).detach()
        real_output = model.discriminator(segmented_images, real_images)
        fake_output = model.discriminator(segmented_images, fake_images)
        d_loss = model.discriminator_loss(real_output, fake_output)
        
        d_loss.backward()
        opt_d.step()
        
        # 2. Train Generator
        opt_g.zero_grad()
        fake_images = model.generator(segmented_images)
        fake_output = model.discriminator(segmented_images, fake_images)
        g_loss, adv_loss, l1 = model.generator_loss(fake_output, fake_images, real_images)
        
        g_loss.backward()
        opt_g.step()
        
        g_loss_total += g_loss.item()
        d_loss_total += d_loss.item()
        
        loop.set_postfix(
            G_Loss=f"{g_loss.item():.4f} (Adv: {adv_loss.item():.4f}, L1: {l1.item():.4f})", 
            D_Loss=f"{d_loss.item():.4f}"
        )
        
    return g_loss_total / len(loader), d_loss_total / len(loader)

def validate_one_epoch(model, loader):
    """Performs a single validation epoch."""
    model.eval()
    g_loss_total = 0.0
    d_loss_total = 0.0
    
    loop = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for segmented_images, real_images in loop:
            segmented_images = segmented_images.to(config.DEVICE)
            real_images = real_images.to(config.DEVICE)
            
            fake_images = model.generator(segmented_images)
            real_output = model.discriminator(segmented_images, real_images)
            fake_output = model.discriminator(segmented_images, fake_images)
            
            d_loss = model.discriminator_loss(real_output, fake_output)
            g_loss, _, _ = model.generator_loss(fake_output, fake_images, real_images)
            
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

            loop.set_postfix(G_Loss=f"{g_loss.item():.4f}", D_Loss=f"{d_loss.item():.4f}")

    return g_loss_total / len(loader), d_loss_total / len(loader)

def main(args):
    """Main execution function."""
    start_time = time.time()
    
    # Set seed
    torch.manual_seed(config.SEED)
    
    logger.info(f"Starting Pix2Pix training...")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Configuration: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}, Lambda_L1={args.lambda_l1}")
    
    # Create output directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    
    # 1. Load Data
    try:
        train_loader, val_loader = get_loaders(
            train_dir=config.TRAIN_DIR,
            val_dir=config.VAL_DIR,
            batch_size=args.batch_size,
            img_size=config.IMG_SIZE
        )
        if train_loader is None:
            raise FileNotFoundError
        logger.info(f"Data loaded: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")
    except FileNotFoundError:
        logger.error(f"Dataset not found in '{config.DATASET_ROOT}'.")
        logger.error("Please run 'bash setup_dataset.sh' first.")
        return
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        return

    # 2. Initialize Models
    generator = Generator(in_channels=config.CHANNELS_IMG, out_channels=config.CHANNELS_IMG).to(config.DEVICE)
    discriminator = Discriminator(in_channels=config.CHANNELS_IMG).to(config.DEVICE)
    pix2pix = Pix2PixModel(generator, discriminator).to(config.DEVICE)
    
    # 3. Initialize Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(config.BETA1, config.BETA2))
    
    # 4. Training Loop
    history = {
        "train_g_loss": [], "train_d_loss": [],
        "val_g_loss": [], "val_d_loss": []
    }
    
    # Get one batch from val_loader for consistent sample visualization
    sample_val_loader = DataLoader(val_loader.dataset, batch_size=3, shuffle=True)

    try:
        for epoch in range(1, args.epochs + 1):
            train_g_loss, train_d_loss = train_one_epoch(pix2pix, train_loader, g_optimizer, d_optimizer)
            val_g_loss, val_d_loss = validate_one_epoch(pix2pix, val_loader)
            
            history["train_g_loss"].append(train_g_loss)
            history["train_d_loss"].append(train_d_loss)
            history["val_g_loss"].append(val_g_loss)
            history["val_d_loss"].append(val_d_loss)
            
            logger.info(f"Epoch {epoch}/{args.epochs} | G Train Loss: {train_g_loss:.4f} | D Train Loss: {train_d_loss:.4f} | G Val Loss: {val_g_loss:.4f} | D Val Loss: {val_d_loss:.4f}")
            
            if epoch % config.DISPLAY_INTERVAL == 0 or epoch == args.epochs:
                logger.info(f"Saving sample images for epoch {epoch}...")
                save_sample_images(pix2pix, sample_val_loader, epoch, num_samples=3)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving current state.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        return
    finally:
        # 5. Save Results
        logger.info("Training finished. Saving final models and plots.")
        
        # Save checkpoints
        save_checkpoint(generator, g_optimizer, f"{config.CHECKPOINT_DIR}/generator_final.pth")
        save_checkpoint(discriminator, d_optimizer, f"{config.CHECKPOINT_DIR}/discriminator_final.pth")
        
        # Save loss plots
        plot_and_save_losses(history["train_g_loss"], history["train_d_loss"], "Training Loss", "train_loss.png")
        plot_and_save_losses(history["val_g_loss"], history["val_d_loss"], "Validation Loss", "val_loss.png")
        
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Pix2Pix model.")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help=f"Number of training epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help=f"Learning rate (default: {config.LEARNING_RATE})")
    parser.add_argument("--lambda-l1", type=int, default=config.LAMBDA_L1, help=f"Weight for L1 loss (default: {config.LAMBDA_L1})")
    parser.add_argument("--data-dir", type=str, default=config.DATASET_ROOT, help=f"Root dataset directory (default: {config.DATASET_ROOT})")
    
    args = parser.parse_args()
    
    # Update config based on args (optional, but good practice)
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.LAMBDA_L1 = args.lambda_l1
    config.DATASET_ROOT = args.data_dir
    config.TRAIN_DIR = f"{args.data_dir}/train"
    config.VAL_DIR = f"{args.data_dir}/val"
    
    main(args)

# Image-to-Image Translation with Pix2Pix

A PyTorch implementation of the Pix2Pix model (cGAN) for translating semantic segmentation maps to photorealistic images. This project, developed for a graduate-level Generative Models course, explores the U-Net generator and PatchGAN discriminator to learn a mapping from labels to images on the Cityscapes dataset.

> **Note:** A simpler, more focused implementation guide for this project is also available on Medium. You can read it here: [Pix2Pix PyTorch Implementation: What Is It and How to Do It?](https://medium.com/@ms.maryamrezaee/pix2pix-pytorch-implementation-what-is-it-and-how-to-do-it-f53bce51c84e)
> 
## Features

* **U-Net Generator:** Modular U-Net architecture with skip-connections to preserve low-level structural details.
* **PatchGAN Discriminator:** A fully convolutional discriminator that classifies $N \times N$ patches to promote high-frequency realism.
* **Conditional GAN (cGAN):** The generation process is conditioned on an input semantic map.
* **Combined L1 + Adversarial Loss:** Balances pixel-level accuracy with photorealism.
* **Scripted Pipeline:** Includes scripts for dataset setup, training, and visualization.
* **Robust Training:** Features command-line arguments, comprehensive logging (to console and file), and checkpointing.

## Core Concepts & Techniques

* **Generative Adversarial Networks (GANs):** Training a generator and discriminator in a minimax game to learn a data distribution.
* **Conditional GANs (cGANs):** Extending GANs to learn a mapping from an input condition $x$ (label map) to an output $y$ (photo).
* **Image-to-Image Translation:** The task of transforming a representation of a scene from one domain to another.
* **Encoder-Decoder Architecture:** A network design that first compresses information into a latent bottleneck and then reconstructs it.
* **Skip-Connections:** Linking encoder layers directly to decoder layers, preventing information loss and enabling precise translations.

---

## How It Works

This project implements the Pix2Pix framework to learn a mapping $G: x \rightarrow y$, where $x$ is a semantic segmentation map and $y$ is a corresponding photorealistic image.

### 1. Core Architecture

The model is composed of two key networks that are trained simultaneously:

* **Generator (U-Net):** The generator's task is to create a realistic image $G(x)$ that matches the input map $x$. It uses a **U-Net** architecture:
    1.  **Encoder:** A series of convolutional `DownSample` blocks progressively reduce the spatial dimensions (e.g., $256 \times 256 \rightarrow 1 \times 1$), capturing the high-level context of the scene.
    2.  **Decoder:** A series of `UpSample` (ConvTranspose) blocks progressively increase the spatial dimensions, reconstructing the image.
    3.  **Skip-Connections:** The key feature. Each encoder layer $i$ is concatenated with the corresponding decoder layer $n-i$. This allows low-level information (like edges and object boundaries) from the input map to "skip" the bottleneck and be used directly during reconstruction, which is critical for preserving the detailed structure.

* **Discriminator (PatchGAN):** The discriminator's task is to distinguish between real image pairs $(x, y)$ and fake pairs $(x, G(x))$.
    * Instead of classifying the *entire* image as real or fake (which is computationally expensive and can miss local detail), the **PatchGAN** is a fully convolutional network that outputs an $N \times N$ grid.
    * Each cell in this grid represents the "realness" verdict for a corresponding patch (e.g., $70 \times 70$) of the input.
    * This forces the generator to produce realistic high-frequency details (textures, sharp edges) across the *entire* image, not just a plausible-looking result.

### 2. Algorithms & Loss Functions

The training process is a "minimax game" governed by two loss functions that are combined for the generator.

* **Adversarial Loss (cGAN Loss):**

  This is the standard GAN loss, conditioned on the input $x$. The generator $G$ tries to minimize this loss (fool $D$), while the discriminator $D$ tries to maximize it.

  $$\mathcal{L}\_{GAN}(G, D) = \mathbb{E}\_{x, y}[\log D(x, y)] + \mathbb{E}_{x}[\log (1 - D(x, G(x)))]$$

* **L1 Reconstruction Loss:**

  The adversarial loss alone can produce realistic images that are not structurally faithful to the input map. To fix this, an **L1 Loss** (Mean Absolute Error) is added, which forces $G(x)$ to be pixel-wise close to the ground truth $y$.

  $$\mathcal{L}\_{L1}(G) = \mathbb{E}_{x, y} \| y - G(x) \|_1$$

* **Final Objective:**

  The generator's final loss is a weighted sum of both, where $\lambda$ (set to 100) gives high importance to the L1 reconstruction.

  $$G^* = \arg \min_G \max_D \mathcal{L}\_{GAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$$

### 3. Analysis of Results

Based on the training run (50 epochs, batch size 16):

* **Image Quality:** The model successfully learns the mapping from semantic maps to images. The generated images preserve the high-level structure (e.g., road placement, tree-lines) and the correct color palettes (e.g., green for vegetation, grey for roads).
* **Blurriness:** The generated images exhibit some blurriness and lack fine-grained texture. This is a known side-effect of the strong **L1 loss**, which encourages the generator to find a "safe" or "average" pixel value that minimizes pixel-wise error, resulting in a less sharp image. The original paper trained for 200 epochs; with more training, the adversarial loss would continue to push the generator to produce sharper, more realistic high-frequency details to "fool" the PatchGAN discriminator.
* **Loss Curves:** The training and validation loss curves show a stable process.
    * The **Generator Loss** consistently decreases, showing it is successfully learning.
    * The **Discriminator Loss** stabilizes around 0.5-0.7. This is a sign of a healthy equilibrium: the discriminator is not overpowering the generator (which would cause its loss to drop to 0) and is not failing to learn (which would cause its loss to rise), allowing both models to improve together.

---

## Project Structure

```
pytorch-pix2pix-image-translation/
├── .gitignore              # Ignores data, logs, outputs, and pycache
├── LICENSE                 # MIT License file
├── README.md               # This project README
├── requirements.txt        # Python dependencies
├── setup_dataset.sh        # Bash script to download and unzip the dataset
├── train.py                # Main training script (entry point)
├── run_pix2pix.ipynb       # Jupyter Notebook for explanation, execution, and analysis
└── src/
    ├── __init__.py         # Makes src/ a Python module
    ├── config.py           # Stores all hyperparameters and constants
    ├── dataset.py          # CityscapesDataset class and DataLoader functions
    ├── models.py           # Generator (U-Net) and Discriminator (PatchGAN)
    ├── pix2pix_model.py    # Wrapper class combining G & D with loss functions
    └── utils.py            # Helper functions (logging, checkpoints, plotting)
````

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/pytorch-pix2pix-image-translation.git
    cd pytorch-pix2pix-image-translation
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup the Data:**
    You must have a `kaggle.json` API key file in the root `pytorch-pix2pix-image-translation/` directory. You can create one from your Kaggle account settings.
    ```bash
    # Make the script executable and run it
    chmod +x setup_dataset.sh
    ./setup_dataset.sh
    ```
    This will create a `pytorch-cityscapes_pix2pix_dataset/` folder containing the `train/` and `val/` sub-directories.

4.  **Run the Training Script:**
    You can run the training directly from the command line. All outputs will be saved to the `outputs/` directory and logs to `logs/`.
    ```bash
    python train.py --epochs 50 --batch-size 16 --lr 0.0002
    ```
    * `--epochs`: Number of epochs to train (default: 50).
    * `--batch-size`: Training batch size (default: 16).
    * `--lr`: Learning rate (default: 0.0002).

5.  **View Results:**
    * Sample generated images are saved in `outputs/samples/`.
    * Loss plots are saved in `outputs/plots/`.
    * Final model checkpoints are saved in `outputs/checkpoints/`.
    * A detailed log file is available at `logs/pix2pix.log`.

3.  **Example Usage / Guided Walkthrough:**
    For a comprehensive, step-by-step walkthrough that includes the detailed theory, code execution, and analysis of the final results, please open and run the `run_pix2pix.ipynb` notebook.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

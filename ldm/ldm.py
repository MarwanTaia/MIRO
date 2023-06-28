###################################################################################################
# Imports
###################################################################################################
# Local
from utils import *
from train_vqgan import train_vqgan
from train_ddpm import train_ddpm
# General
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pynvml import *
# PyTorch
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# MONAI
import monai
from monai.utils import first
from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset
from generative.networks.nets import (DiffusionModelUNet, PatchDiscriminator, VQVAE)
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer, LatentDiffusionInferer


###################################################################################################
# Functions
###################################################################################################
def plot_example_images(util_image, num_example_images, image_size, figure_dir):
    fig, ax = plt.subplots(6, num_example_images, figsize=(20, num_example_images))
    fig.set_dpi(1000)
    fig.suptitle(f"Example CT scan from training set ({image_size[0]}x{image_size[1]}x{image_size[2]})", fontsize=16)

    x_lin = np.linspace(0, util_image.shape[-1] - 1, num_example_images).astype(int)
    y_lin = np.linspace(0, util_image.shape[-2] - 1, num_example_images).astype(int)
    z_lin = np.linspace(0, util_image.shape[-3] - 1, num_example_images).astype(int)

    for i in range(num_example_images):
        ax[0, i].imshow(util_image[0, 0, z_lin[i], :, :].cpu().numpy(), cmap="gray")
        ax[1, i].imshow(util_image[0, 1, z_lin[i], :, :].cpu().numpy())
        ax[0, i].set_title(f"Slice : {z_lin[i]+1}")
        ax[2, i].imshow(util_image[0, 0, :, y_lin[i], :].cpu().numpy(), cmap="gray")
        ax[3, i].imshow(util_image[0, 1, :, y_lin[i], :].cpu().numpy())
        ax[4, i].imshow(util_image[0, 0, :, :, x_lin[i]].cpu().numpy(), cmap="gray")
        ax[5, i].imshow(util_image[0, 1, :, :, x_lin[i]].cpu().numpy())

        ax[0, 0].set_ylabel("Horizontal")
        ax[2, 0].set_ylabel("Coronal")
        ax[4, 0].set_ylabel("Sagittal")

        for j in range(6):
            ax[j, i].set_yticks([])
            ax[j, i].set_xticks([])

    fig.subplots_adjust(top=0.85, wspace=1)
    plt.savefig(os.path.join(figure_dir, "train_example.pdf"), format="pdf", bbox_inches="tight")


###################################################################################################
# Settings
###################################################################################################
# Set the multiprocessing method to use the file system (avoids issues with fork during training)
torch.multiprocessing.set_sharing_strategy('file_system')

# Set GPU device
device, device_ids = set_device(verbose=True)

# Print GPU memory
# get_gpu_mem(message="GPU memory usage before loading data:", device_ids=device_ids)

# Set random seed
seed = 20301
torch.manual_seed(seed)
np.random.seed(seed)

# Set directories
## Data directory
data_dir = "/home/taia/Téléchargements/"
## Data CSV file
data_csv = os.path.join(data_dir, "data_index.csv")
## Resources directory
resource = "Task01_BrainTumour.tar"
## Output directory
output_dir = "/home/taia/junk/remote_test/"
## Compressed file directory
compressed_file = os.path.join(data_dir, resource)
## Checkpoint directory
checkpoint_dir = os.path.join(output_dir, "checkpoints")
## Log directory
log_dir = os.path.join(output_dir, "logs")
## Figure directory
figure_dir = os.path.join(output_dir, "figures")
## VQ-GAN checkpoint path
vqgan_checkpoint_path = os.path.join(checkpoint_dir, "vqgan_checkpoint.pt")
## Diffusion checkpoint path
ddpm_checkpoint_path = os.path.join(checkpoint_dir, "ddpm_checkpoint.pt")
## Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# Set parameters
## Number of epochs
epochs = 1000 # 250 # 1000
## Validation interval
valid_interval = 5 # 20 # 50
## Batch size
batch_size = 1
## Number of workers
num_workers = 16
## Learning rate
lr = 1e-3
## Number of diffusion steps
num_diffusion_steps = 1000
## Image size
image_size = (240, 240, 120)
## Volume size
voxel_size = (2, 2, 2)
## Number of example images
num_example_images = 7
## Latent channels
latent_channels = 3 # 256 # Try 512/64 32/512
## DDPM channels
ddpm_channels = 8
## Embedding dimension
embedding_dim = 8
## Codebook size
num_embeddings = 1024*8*4 # 128
## MRI channel
mri_channel = 0 # 0: FLAIR
## Label channel
label_channel = 1


###################################################################################################
# Main
###################################################################################################
def main():

    ###############################################################################################
    # Data Loading
    ###############################################################################################
    # Create transforms
    loading_transforms = create_transforms(image_size)

    # Create training dataset and loader
    train_dataset = create_dataset(data_dir, "training", loading_transforms, seed=0)
    train_img_list, train_label_list = create_img_label_list(
        train_dataset, mri_channel, label_channel
    )
    train_combined_tensor = create_combined_tensor(train_img_list, train_label_list)
    train_loader = create_data_loader(
        train_combined_tensor, 
        batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    # Create validation dataset and loader
    valid_dataset = create_dataset(data_dir, "validation", loading_transforms, seed=0)
    valid_img_list, valid_label_list = create_img_label_list(
        valid_dataset, mri_channel, label_channel
    )
    valid_combined_tensor = create_combined_tensor(valid_img_list, valid_label_list)
    valid_loader = create_data_loader(
        valid_combined_tensor, 
        batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    util_image = first(train_loader)
    original_shape = util_image.shape

    ###############################################################################################

    # Load first image of the training set
    util_image = first(train_loader)
    print(f"Image shape: {util_image.shape}")

    plot_example_images(util_image, num_example_images, image_size, figure_dir)

    ###############################################################################################
    # Model : VQ-GAN
    ###############################################################################################
    # Instantiate the model
    vqvae = VQVAE(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        downsample_parameters=((2, 4, 1, 1),    (2, 4, 1, 1),    (2, 4, 1, 1)),
        upsample_parameters=(  (2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),

        num_res_layers=3,
        num_channels=[latent_channels, latent_channels, latent_channels],
        num_res_channels=[latent_channels, latent_channels, latent_channels],

        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
    ).to(device)

    # Instantiate the discriminator
    discriminator = PatchDiscriminator(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        num_layers_d=4,
        num_channels=4,
    ).to(device)

    # Define the loss functions
    # The perceptual loss is used to train the VQ-VAE
    perceptual_loss = PerceptualLoss(
        spatial_dims=3,
        is_fake_3d=False,
        network_type="medicalnet_resnet10_23datasets"
        # network_type="vgg",
    ).to(device)

    # The adversarial loss is used to train the discriminator
    l1_loss = torch.nn.L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    adv_weight = 0.01
    perceptual_weight = 0.001

    # Define optimizers
    optimizer_g = torch.optim.Adam(vqvae.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    #################################################
    # Reload model
    #################################################
    # Initialize elements
    start_epoch = 0
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []
    best_valid_loss = np.inf

    # Load model
    # start_epoch, epoch_recon_loss_list, epoch_gen_loss_list, epoch_disc_loss_list, \
    #     val_recon_epoch_loss_list, intermediary_images, \
    #     best_valid_loss = load_checkpoint_vqvae(
    #     vqgan_checkpoint_path,
    #     vqvae,
    #     discriminator,
    #     perceptual_loss,
    #     optimizer_g,
    #     optimizer_d
    #     )

    ###################################################################################################
    # Training : VQ-GAN
    ###################################################################################################

    train_vqgan(
        train_loader=train_loader,
        valid_loader=valid_loader,
        vqvae=vqvae,
        discriminator=discriminator,
        reconstructions_loss=l1_loss,
        perceptual_loss=perceptual_loss,
        adv_loss=adv_loss,
        perceptual_weight=perceptual_weight,
        adv_weight=adv_weight,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        device=device,
        start_epoch=start_epoch,
        epochs=epochs,
        valid_interval=valid_interval,
        num_example_images=num_example_images,
        original_shape=original_shape,
        figure_dir=figure_dir,
        vqgan_checkpoint_path=vqgan_checkpoint_path,
        plot=True,
    )

    ############################################################################################################
    # Model : LDM
    ############################################################################################################

    # clear gpu data memory
    torch.cuda.empty_cache()

    # Diffusion Model
    ddpm = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=embedding_dim,
        out_channels=embedding_dim,

        num_res_blocks=[2],
        num_channels=[ddpm_channels],
        resblock_updown=True,

        attention_levels=[False],
        num_head_channels=8,
        with_conditioning=False,
        transformer_num_layers=0, #1
        # cross_attention_dim=0, #16
        upcast_attention=True,

        norm_num_groups=1
    ).to(device)

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
    )

    # Caclulate scaling factor
    with torch.no_grad():
        with torch.autocast(enabled=True, device_type="cuda"):
            z = vqvae.encode_stage_2_inputs(util_image.to(device))

    print(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    # LDM Inferer
    ldm_inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    # DDPM inferer
    ddpm_inferer = DiffusionInferer(scheduler=scheduler)

    # Optimizer
    optimizer_diff = torch.optim.Adam(params=ddpm.parameters(), lr=lr)

    ###############################################################################################
    # Training : LDM
    ###############################################################################################

    epoch_loss_list = []
    vqvae.eval()
    scaler = GradScaler()
    val_epoch_loss_list = []
    best_valid_loss = float("inf")

    start_epoch = 0
    start_epoch, epoch_loss_list, val_epoch_loss_list, best_valid_loss = load_checkpoint_ddpm(
        ddpm_checkpoint_path, ddpm, optimizer_diff)


    # first_batch = first(data_loader)
    z = vqvae.encode_stage_2_inputs(util_image.to(device))

    train_ddpm(
        train_loader=train_loader,
        valid_loader=valid_loader,
        vqvae=vqvae,
        ddpm=ddpm,
        ldm_inferer=ldm_inferer,
        ddpm_inferer=ddpm_inferer,
        optimizer_diff=optimizer_diff,
        scaler=scaler,
        device=device,
        start_epoch=start_epoch,
        epochs=epochs,
        valid_interval=valid_interval,
        num_example_images=num_example_images,
        original_shape=original_shape,
        figure_dir=figure_dir,
        ddpm_checkpoint_path=ddpm_checkpoint_path,
        plot=True,
    )

if __name__ == "__main__":
    main()

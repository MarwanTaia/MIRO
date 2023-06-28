###################################################################################################
# Imports
###################################################################################################
# General imports
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
from pynvml import *
# MONAI imports
from monai.config import print_config
from monai.utils import first
from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset, DistributedSampler
from monai.handlers import CheckpointSaver, CheckpointLoader
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator, VQVAE
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer, LatentDiffusionInferer
import monai

###################################################################################################
# Settings
###################################################################################################
# Set the multiprocessing method to use the file system (avoids issues with fork during training)
torch.multiprocessing.set_sharing_strategy('file_system')
# dist.init_process_group(backend='nccl')

# Set GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Devices available: {torch.cuda.device_count()}")
print(f"Devices indexes: {device.index}")
device_ids = [i for i in range(torch.cuda.device_count())]

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
epochs = 2 # 250
## Validation interval
valid_interval = 1 # 20
## Batch size
batch_size = 1
## Number of workers
num_workers = 0
## Learning rate
lr = 1e-4
## Number of diffusion steps
num_diffusion_steps = 1000
## Image size
image_size = (120, 120, 120)
## Volume size
voxel_size = (2, 2, 2)
## Number of example images
num_example_images = 7
## Latent channels
latent_channels = 32 # 256 # Try 512/64 32/512
## Codebook size
num_embeddings = 1024*4 # 128
## MRI channel
mri_channel = 0 # 0: FLAIR

###################################################################################################
# Functions
###################################################################################################
def get_gpu_mem(message='', device_ids=[0]):
    '''
    FUNCTION:
    ---------
        get_gpu_mem
    DESCRIPTION:
    ------------
        Get and print the GPU memory usage along with a message, if provided.
    ARGUMENTS:
    ----------
        Message: str
    RETURNS:
    --------
        None
    '''
    nvmlInit()
    print(message)
    for i in device_ids:
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        print(f'GPU {i}:')
        print(f'total (GB): {info.total / 1024 / 1024 / 1024}')
        print(f'free  (GB): {info.free / 1024 / 1024 / 1024}')
        print(f'used  (GB): {info.used / 1024 / 1024 / 1024}')


def load_checkpoint_vqvae(model_checkpoint_dir, start_epoch, epoch_recon_loss_list,
                          epoch_gen_loss_list, epoch_disc_loss_list, val_recon_epoch_loss_list,
                          intermediary_images, best_valid_loss):
    if os.path.exists(model_checkpoint_dir):
        checkpoint = torch.load(model_checkpoint_dir)
        vqvae.load_state_dict(checkpoint['vqvae'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        perceptual_loss.load_state_dict(checkpoint['perceptual_loss'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        epoch_recon_loss_list = checkpoint['epoch_recon_loss_list']
        epoch_gen_loss_list = checkpoint['epoch_gen_loss_list']
        epoch_disc_loss_list = checkpoint['epoch_disc_loss_list']
        val_recon_epoch_loss_list = checkpoint['val_recon_epoch_loss_list']
        intermediary_images = checkpoint['intermediary_images']
        start_epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']

        print(f"Checkpoint loaded. Starting from epoch {start_epoch + 1}")
    else:
        print("No previous checkpoint found. Training from scratch.")
    return start_epoch, epoch_recon_loss_list, epoch_gen_loss_list, epoch_disc_loss_list, val_recon_epoch_loss_list, intermediary_images, best_valid_loss


def load_checkpoint_ddpm(model_checkpoint_dir, start_epoch, epoch_loss_list, val_epoch_loss_list, best_valid_loss):
    if os.path.exists(model_checkpoint_dir):
        checkpoint = torch.load(model_checkpoint_dir)
        ddpm.load_state_dict(checkpoint["ddpm"])
        optimizer_diff.load_state_dict(checkpoint["optimizer_diff"])
        start_epoch = checkpoint["epoch"]
        epoch_loss_list = checkpoint["epoch_loss_list"]
        val_epoch_loss_list = checkpoint["val_epoch_loss_list"]
        best_valid_loss = checkpoint["best_valid_loss"]

        print(f"Checkpoint loaded. Starting from epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting from scratch.")
    return start_epoch, epoch_loss_list, val_epoch_loss_list, best_valid_loss


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


def reconstruction_comparison_plot(figure_dir, figure_name, images, reconstruction, num_example_images):
    x_lin = np.linspace(0, images.shape[-1] - 1, num_example_images).astype(int)
    y_lin = np.linspace(0, images.shape[-2] - 1, num_example_images).astype(int)
    z_lin = np.linspace(0, images.shape[-3] - 1, num_example_images).astype(int)
    fig, ax = plt.subplots(4, num_example_images, figsize=(20, 5))
    for i in range(num_example_images):
        ax[0, i].imshow(images[0, 0, z_lin[i], :, :].cpu().numpy(), cmap="gray")
        ax[0, i].axis("off")
        ax[0, i].set_title(f"Slice {z_lin[i]+1}")
        ax[1, i].imshow(reconstruction[0, 0, z_lin[i], :, :].cpu().numpy(), cmap="gray")
        ax[1, i].axis("off")
        ax[2, i].set_title(f"Slice {z_lin[i]}")
        ax[2, i].imshow(images[0, 1, z_lin[i], :, :].cpu().numpy())
        ax[2, i].axis("off")
        ax[3, i].imshow(reconstruction[0, 1, z_lin[i], :, :].cpu().numpy())
        ax[3, i].axis("off")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.png"))


def reconstruction_loss_plot(figure_dir, figure_name, epoch_recon_loss_list, val_recon_epoch_loss_list, valid_interval):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_recon_loss_list, label="train")
    plt.plot(np.arange(0, len(epoch_recon_loss_list), valid_interval), val_recon_epoch_loss_list, label="validation")
    plt.legend()
    plt.title("Reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), format="pdf", bbox_inches="tight")


def adversarial_loss_plot(figure_dir, figure_name, epoch_gen_loss_list, epoch_disc_loss_list, valid_interval):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_gen_loss_list, label="generator")
    plt.plot(epoch_disc_loss_list, label="discriminator")
    plt.legend()
    plt.title("Generator and discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), format="pdf", bbox_inches="tight")


def latent_representation_plot(figure_dir, figure_name, latent_img, num_example_images):
    # Checking latent space
    fig, ax = plt.subplots(3, num_example_images, figsize=(20, num_example_images))
    # Set dpi 
    fig.set_dpi(1000)
    # Set main title
    fig.suptitle("Example of a Latent Representation of CT scan", fontsize=16)
    # Get linspace for the z axis
    x_lin = np.linspace(0, latent_img.shape[-1] - 1, num_example_images).astype(int)
    y_lin = np.linspace(0, latent_img.shape[-2] - 1, num_example_images).astype(int)
    z_lin = np.linspace(0, latent_img.shape[-3] - 1, num_example_images).astype(int)
    # Plot the images
    for i in range(num_example_images):
        ax[0, i].imshow(latent_img[0, 0, z_lin[i], :, :].cpu().numpy(), cmap="gray")
        ax[0, i].set_title(f"Slice : {x_lin[i]}")
        ax[1, i].imshow(latent_img[0, 0, :, y_lin[i], :].cpu().numpy(), cmap="gray")
        ax[2, i].imshow(latent_img[0, 0, :, :, x_lin[i]].cpu().numpy(), cmap="gray")
        # Set row titles
        ax[0, 0].set_ylabel("Horizontal")
        ax[1, 0].set_ylabel("Coronal")
        ax[2, 0].set_ylabel("Sagittal")
    # Set axis off
    for i in range(3):
        for j in range(num_example_images):
            ax[i, j].set_yticks([])
            ax[i, j].set_xticks([])
    # Save figure
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), format="pdf", bbox_inches="tight")


def ldm_sampling_plot(figure_dir, figure_name, images, noise, latent_image, denoised_image, out_image, num_example_images):
    # Checking denoising
    x_lin = np.linspace(0, noise.shape[-1] - 1, num_example_images).astype(int)
    y_lin = np.linspace(0, noise.shape[-2] - 1, num_example_images).astype(int)
    z_lin_pixel = np.linspace(0, images.shape[-3] - 1, num_example_images).astype(int)
    z_lin_latent = np.linspace(0, noise.shape[-3] - 1, num_example_images).astype(int)

    fig, ax = plt.subplots(6, num_example_images, figsize=(20, 10))
    # Set dpi 
    fig.set_dpi(1000)
    fig.suptitle("Example of LDM Process Steps", fontsize=16)
    for i in range(num_example_images):
        ax[0, i].imshow(images[0, 0, z_lin_pixel[i], :, :].cpu().numpy(), cmap="gray")
        ax[0, i].set_title(f"Slice : {z_lin_pixel[i]+1}", fontsize=12)
        ax[1, i].imshow(latent_image[0, 0, z_lin_latent[i], :, :].cpu().numpy(), cmap="gray")
        ax[1, i].set_title(f"Slice : {z_lin_latent[i]+1}", fontsize=12)
        ax[2, i].imshow(noise[0, 0, z_lin_latent[i], :, :].cpu().numpy(), cmap="gray")
        ax[3, i].imshow(denoised_image[0, 0, z_lin_latent[i], :, :].cpu().numpy(), cmap="gray")
        ax[4, i].imshow(out_image[0, 0, z_lin_pixel[i], :, :].cpu().numpy(), cmap="gray")
        ax[4, i].set_title(f"Slice : {z_lin_pixel[i]+1}", fontsize=12)
        ax[5, i].imshow(out_image[0, 1, z_lin_pixel[i], :, :].cpu().numpy())
        for j in range(6):
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
    # Set row titles
    ax[0, 0].set_ylabel("Original Image \n (Pixel Space)", fontsize=12)
    ax[1, 0].set_ylabel("Latent \n Representation\n (Latent Space)", fontsize=12)
    ax[2, 0].set_ylabel("Noise \n (Latent Space)", fontsize=12)
    ax[3, 0].set_ylabel("Denoised Image \n (Latent Space)", fontsize=12)
    ax[4, 0].set_ylabel("Output Image \n (Pixel Space)", fontsize=12)
    ax[5, 0].set_ylabel("Output Mask \n (Pixel Space)", fontsize=12)
    # Limit space between plots
    fig.subplots_adjust(wspace=-0.7, hspace=0.23, top=0.9)
    # Save figure
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), format="pdf", bbox_inches="tight")


def ddpm_loss_plot(figure_dir, figure_name, epoch_loss_list, val_epoch_loss_list, valid_interval, epoch):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_loss_list, label="train")
    plt.plot(np.arange(0, epoch + 1, valid_interval), val_epoch_loss_list, label="validation")
    plt.legend()
    plt.title("Denoising (DDPM) Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(figure_dir, f"{figure_name}.pdf"), format="pdf", bbox_inches="tight")


###################################################################################################
# Data Loading
###################################################################################################
# get_gpu_mem('Before loading data', device_ids)

# Define the transforms
transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.EnsureTyped(keys=["image", "label"]),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        # transforms.Lambdad(keys="image", func=lambda x: x[mri_channel, :, :, :]),
        # transforms.AddChanneld(keys=["image"]),
        # transforms.EnsureTyped(keys=["image"]),
        # transforms.Spacingd(keys=["image"], pixdim=voxel_size, mode=("bilinear")),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 2)),
        transforms.Resized(keys=["image", "label"], spatial_size=image_size),
        # transforms.CenterSpatialCropd(keys=["image"], roi_size=image_size),
        transforms.NormalizeIntensityd(keys=["image"]),
        # transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.ToTensord(keys=["image", "label"]),
    ]
)

# Create a dataset
dataset = DecathlonDataset(
    root_dir=data_dir,
    task="Task01_BrainTumour",
    section="training",
    num_workers=num_workers,
    download=False,
    seed=0,
    transform=transforms,
    val_frac=0.99
)

# Convert the dataset images to a list of tensors of one of the channels
train_img_list = [dataset[i]['image'][mri_channel, :, :, :] for i in range(len(dataset))]
print(f"Length of the list of tensors: {len(train_img_list)}")
print(f"Shape of the first image tensor: {train_img_list[0].shape}")
# Convert the dataset lables to a list of tensors of one of the channels
label_channel = 1
train_label_list = [dataset[i]['label'][label_channel, :, :, :] for i in range(len(dataset))]
print(f"Length of the list of tensors: {len(train_label_list)}")
print(f"Shape of the first label tensor: {train_label_list[0].shape}")
# Convert the list of tensors to a single tensor
train_img = torch.stack(train_img_list)
train_label = torch.stack(train_label_list)
print(f"Shape of the image tensor: {train_img.shape}")
print(f"Shape of the label tensor: {train_label.shape}")
# Concatenate the images and labels. Each volume is combined with its corresponding label
# So that the first channel is the image and the second channel is the label
# Add channel dimension to the image and label tensors
train_img = train_img.unsqueeze(1)
train_label = train_label.unsqueeze(1)
print(f"Shape of the image tensor after adding channel dimension: {train_img.shape}")
print(f"Shape of the label tensor after adding channel dimension: {train_label.shape}")
# Combine the image and label tensors
combined_tensor = torch.cat((train_img, train_label), dim=1)
print(f"Shape of the combined tensor: {combined_tensor.shape}")
print(f"Shape of the first volume: {combined_tensor[0].shape}")

# Create a dataset
train_ds = Dataset(data=combined_tensor)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)

# Create a dataset
dataset = DecathlonDataset(
    root_dir=data_dir,
    task="Task01_BrainTumour",
    section="validation",
    num_workers=num_workers,
    download=False,
    seed=0,
    transform=transforms,
    val_frac=0.01
)

# Convert the dataset images to a list of tensors of one of the channels
train_img_list = [dataset[i]['image'][mri_channel, :, :, :] for i in range(len(dataset))]
print(f"Length of the list of tensors: {len(train_img_list)}")
print(f"Shape of the first image tensor: {train_img_list[0].shape}")
# Convert the dataset lables to a list of tensors of one of the channels
label_channel = 1
train_label_list = [dataset[i]['label'][label_channel, :, :, :] for i in range(len(dataset))]
print(f"Length of the list of tensors: {len(train_label_list)}")
print(f"Shape of the first label tensor: {train_label_list[0].shape}")
# Convert the list of tensors to a single tensor
train_img = torch.stack(train_img_list)
train_label = torch.stack(train_label_list)
print(f"Shape of the image tensor: {train_img.shape}")
print(f"Shape of the label tensor: {train_label.shape}")
# Concatenate the images and labels. Each volume is combined with its corresponding label
# So that the first channel is the image and the second channel is the label
# Add channel dimension to the image and label tensors
train_img = train_img.unsqueeze(1)
train_label = train_label.unsqueeze(1)
print(f"Shape of the image tensor after adding channel dimension: {train_img.shape}")
print(f"Shape of the label tensor after adding channel dimension: {train_label.shape}")
# Combine the image and label tensors
combined_tensor = torch.cat((train_img, train_label), dim=1)
print(f"Shape of the combined tensor: {combined_tensor.shape}")
print(f"Shape of the first volume: {combined_tensor[0].shape}")

# Create a dataset
valid_ds = Dataset(data=combined_tensor)
valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=True, num_workers=8)

check_data = first(train_loader)
original_shape = check_data.shape

###############################################################################################

# Load first image of the training set
util_image = first(train_loader)
print(f"Image shape: {util_image.shape}")

plot_example_images(util_image, num_example_images, image_size, figure_dir)

###################################################################################################
# Model : VQ-GAN
###################################################################################################
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
    embedding_dim=latent_channels,
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

# Print GPU memory
# get_gpu_mem("VQ-GAN models loaded", device_ids)

# The adversarial loss is used to train the discriminator
l1_loss = torch.nn.L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01
perceptual_weight = 0.001

# Define optimizers
optimizer_g = torch.optim.Adam(vqvae.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

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
# start_epoch, epoch_recon_loss_list, epoch_gen_loss_list, epoch_disc_loss_list, val_recon_epoch_loss_list, intermediary_images, best_valid_loss = load_checkpoint_vqvae(vqgan_checkpoint_path, start_epoch, epoch_recon_loss_list,
#                           epoch_gen_loss_list, epoch_disc_loss_list, val_recon_epoch_loss_list,
#                           intermediary_images, best_valid_loss)
# print(f"Checkpoint loaded. Starting from epoch {start_epoch + 1}")

# Reload those elements then

###################################################################################################
# Training : VQ-GAN
###################################################################################################

total_start = time.time()
for epoch in range(start_epoch, epochs):
    vqvae.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        torch.cuda.empty_cache()
        images = batch.to(device)
        get_gpu_mem("Batch loaded", device_ids)
        optimizer_g.zero_grad(set_to_none=True)

        encoded = vqvae.encode_stage_2_inputs(images)
        print(f"encoded shape : {encoded.shape}")

        if images.shape != original_shape:
            # Incompatible image, skipping
            print(f"\n\n\nSkipping step {step} due to incompatible image shape\n\n\n")
            continue

        # Generator part
        reconstruction, quantization_loss = vqvae(images)
        print(f"reconstruction shape : {reconstruction.shape}")
        print(f"images shape : {images.shape}")
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]

        recons_loss = l1_loss(reconstruction.float(), images.float())
        recon_p = reconstruction[:, 0, :, :, :].unsqueeze(1)
        images_p = images[:, 0, :, :, :].unsqueeze(1)
        print(f"shape of recon_p : {recon_p.shape}")
        p_loss = perceptual_loss(recon_p.float(), images_p.float())
        generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # Discriminator part
        optimizer_d.zero_grad(set_to_none=True)

        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = adv_weight * discriminator_loss

        loss_d.backward()
        optimizer_d.step()

        epoch_loss += recons_loss.item()
        gen_epoch_loss += generator_loss.item()
        disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )

        # Free memory
        del images, batch, reconstruction
        torch.cuda.empty_cache()

    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
    epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % valid_interval == 0:
        vqvae.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(valid_loader, start=1):
                images = batch.to(device)

                if images.shape != original_shape:
                    # Incompatible image, skipping
                    print(f"\n\n\nSkipping step {step} due to incompatible image shape\n\n\n")
                    continue

                reconstruction, quantization_loss = vqvae(images)

                # get the first sammple from the first validation batch for visualization
                # purposes
                if val_step == 1:
                    intermediary_images.append(reconstruction[:num_example_images, 0])

                recons_loss = l1_loss(reconstruction.float(), images.float())

                val_loss += recons_loss.item()

                # Plot reconstructions
                if val_step == 1:
                    reconstruction_comparison_plot(
                        figure_dir=figure_dir,
                        figure_name=f"reconstruction_comparison_{epoch + 1}",
                        images=images,
                        reconstruction=reconstruction,
                        num_example_images=num_example_images,
                    )

                # Free memory
                del images, batch, reconstruction
                torch.cuda.empty_cache()

        val_loss /= val_step
        val_recon_epoch_loss_list.append(val_loss)

        # Plot losses: reconstruction train and validation
        reconstruction_loss_plot(
            figure_dir=figure_dir,
            figure_name="reconstruction_loss",
            epoch_recon_loss_list=epoch_recon_loss_list,
            val_recon_epoch_loss_list=val_recon_epoch_loss_list,
            valid_interval=valid_interval
        )

        # Plot losses : generator and discriminator
        adversarial_loss_plot(
            figure_dir=figure_dir,
            figure_name="adversarial_loss",
            epoch_gen_loss_list=epoch_gen_loss_list,
            epoch_disc_loss_list=epoch_disc_loss_list,
            valid_interval=valid_interval
        )

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(
                {
                    "vqvae": vqvae.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "perceptual_loss": perceptual_loss.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "epoch_recon_loss_list": epoch_recon_loss_list,
                    "epoch_gen_loss_list": epoch_gen_loss_list,
                    "epoch_disc_loss_list": epoch_disc_loss_list,
                    "val_recon_epoch_loss_list": val_recon_epoch_loss_list,
                    "intermediary_images": intermediary_images,
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss,
                },
                vqgan_checkpoint_path,
            )
            print(f"Checkpoint saved. Best validation loss: {best_valid_loss}")

    # clear gpu data memory
    torch.cuda.empty_cache()

total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")
torch.cuda.empty_cache()

# Plot example reconstruction on an axis, ground truth and reconstruction
# Checking reconstruction
vqvae.eval()
with torch.no_grad():
    image = first(valid_loader)
    image = image.to(device)
    latent_img = vqvae.encode_stage_2_inputs(image)
    reconstruction, quantization_loss = vqvae(images=image)

reconstruction_comparison_plot(
    figure_dir=figure_dir,
    figure_name="reconstruction_comparison_example",
    images=image,
    reconstruction=reconstruction,
    num_example_images=num_example_images,
)

# Plot example latent space on all three axis
latent_representation_plot(
    figure_dir=figure_dir,
    figure_name="latent_representation_example",
    latent_img=latent_img,
    num_example_images=num_example_images,
)
# TODO : Plot latent space of masks ?

############################################################################################################
# Model : LDM
############################################################################################################

# clear gpu data memory
torch.cuda.empty_cache()

# Diffusion Model
ddpm = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=latent_channels,
    out_channels=latent_channels,

    num_res_blocks=[16],
    num_channels=[16],
    resblock_updown=True,

    attention_levels=[True],
    num_head_channels=16,
    with_conditioning=True,
    transformer_num_layers=8, #1
    cross_attention_dim=16, #16
    upcast_attention=True,

    norm_num_groups=1
).to(device)

# Scheduler
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
)

# Caclulate scaling factor
check_data = first(train_loader)
first_batch = check_data
with torch.no_grad():
    with torch.autocast(enabled=True, device_type="cuda"):
        z = vqvae.encode_stage_2_inputs(check_data.to(device))

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)

# LDM Inferer
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
# DDPM inferer
ddpm_inferer = DiffusionInferer(scheduler)

# Optimizer
optimizer_diff = torch.optim.Adam(params=ddpm.parameters(), lr=1e-4)

# Do data parallel
# ddpm = torch.nn.DataParallel(ddpm)


############################################################################################################
# Training : LDM
############################################################################################################

epoch_loss_list = []
vqvae.eval()
scaler = GradScaler()
val_epoch_loss_list = []
best_valid_loss = float("inf")

start_epoch = 0
# start_epoch, epoch_loss_list, val_epoch_loss_list, best_valid_loss = load_checkpoint_ddpm(
#     ddpm_checkpoint_path, start_epoch, epoch_loss_list, val_epoch_loss_list, best_valid_loss)


# first_batch = first(data_loader)
z = vqvae.encode_stage_2_inputs(first_batch.to(device))
original_shape = first_batch.shape

for epoch in range(start_epoch+1, epochs):
    ddpm.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device)
        get_gpu_mem("Batch loaded", device_ids)
        # print(f"images shape : {images.shape}")
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            print(f"shape of z : {z.shape}")
            noise = torch.randn_like(z).to(device)
            noise, _ = vqvae.quantize(noise)
            # we make noise of same type but shape [1, 8, 10, 60, 60]
            # noise = torch.randn((1, 8, 12, 62, 62), dtype=z.dtype, device=z.device)
            get_gpu_mem("Noise loaded", device_ids)
            # print(f"noise shape : {noise.shape}")

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            print(f"shape of images : {images.shape}")
            print(f"shape of noise : {noise.shape}")
            if images.shape != original_shape:
                # Incompatible image, skipping
                print(f"\n\n\nSkipping step {step} due to incompatible image shape\n\n\n")
                continue


            # Get model prediction
            noise_pred = inferer(
                inputs=images, autoencoder_model=vqvae, diffusion_model=ddpm, noise=noise, timesteps=timesteps
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        # Free memory
        del images, noise, noise_pred, loss, timesteps, batch
        torch.cuda.empty_cache()

    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) % valid_interval == 0:
        ddpm.eval()
        with torch.no_grad():
            epoch_loss = 0
            for step, batch in enumerate(valid_loader):
                images = batch.to(device)
                # print(f"images shape : {images.shape}")

                with autocast(enabled=True):
                    # Generate random noise
                    noise = torch.randn_like(z).to(device)
                    noise, _ = vqvae.quantize(noise)
                    # print(f"noise shape : {noise.shape}")

                    # Create timesteps
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                    print(f"shape of images : {images.shape}")
                    print(f"shape of noise : {noise.shape}")
                    if images.shape != original_shape:
                        # Incompatible image, skipping
                        print("Skipping")
                        continue

                    # Get model prediction
                    noise_pred = inferer(
                        inputs=images, autoencoder_model=vqvae, diffusion_model=ddpm, noise=noise, timesteps=timesteps
                    )

                    loss = F.mse_loss(noise_pred.float(), noise.float())

                    if step == 1:
                        with torch.no_grad():
                            latent_img = vqvae.encode_stage_2_inputs(images)
                            noise = torch.randn_like(latent_img).to(device)
                            noise, _ = vqvae.quantize(noise)
                            DDPMinferer = DiffusionInferer(scheduler=scheduler)
                            denoised_image = DDPMinferer.sample(
                                diffusion_model=ddpm,
                                input_noise=noise,
                                scheduler=inferer.scheduler,
                            )
                            out_image = vqvae.decode_stage_2_outputs(denoised_image)

                            
                            ldm_sampling_plot(
                                figure_dir=figure_dir,
                                figure_name=f"ldm_sampling_example_{epoch}",
                                images=images,
                                noise=noise,
                                latent_image=latent_img,
                                denoised_image=denoised_image,
                                out_image=out_image,
                                num_example_images=num_example_images,
                            )

                epoch_loss += loss.item()

            val_epoch_loss_list.append(epoch_loss / (step + 1))

            if epoch_loss / (step + 1) < best_valid_loss:
                best_valid_loss = epoch_loss / (step + 1)
                torch.save(
                    {
                        "ddpm": ddpm.state_dict(),
                        "optimizer_diff": optimizer_diff.state_dict(),
                        "epoch": epoch,
                        "epoch_loss_list": epoch_loss_list,
                        "val_epoch_loss_list": val_epoch_loss_list,
                        "best_valid_loss": best_valid_loss,
                    },
                    ddpm_checkpoint_path,
                )
                print("Checkpoint saved")

            # Free memory
            del images, noise, noise_pred, loss
            torch.cuda.empty_cache()

        # Make loss curves plot
        ddpm_loss_plot(
            figure_dir=figure_dir,
            figure_name=f"ddpm_loss",
            epoch_loss_list=epoch_loss_list,
            val_epoch_loss_list=val_epoch_loss_list,
            valid_interval=valid_interval,
            epoch=epoch,
        )

    # clear gpu data memory
    torch.cuda.empty_cache()
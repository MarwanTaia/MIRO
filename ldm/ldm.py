###################################################################################################
# Imports
###################################################################################################
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


def set_device(type='cuda', verbose=True):
    device = torch.device(type)
    if verbose:
        print(f"Using device: {device}")
        print(f"Devices available: {torch.cuda.device_count()}")
        print(f"Devices indexes: {device.index}")
    device_ids = [i for i in range(torch.cuda.device_count())]
    return device, device_ids


def load_checkpoint_vqvae(model_checkpoint_dir, vqvae, discriminator, perceptual_loss, optimizer_g, optimizer_d):
    if os.path.exists(model_checkpoint_dir):
        checkpoint = torch.load(model_checkpoint_dir)
        vqvae.load_state_dict(checkpoint['vqvae'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        perceptual_loss.load_state_dict(checkpoint['perceptual_loss'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])

        start_epoch = checkpoint['epoch']
        epoch_recon_loss_list = checkpoint['epoch_recon_loss_list']
        epoch_gen_loss_list = checkpoint['epoch_gen_loss_list']
        epoch_disc_loss_list = checkpoint['epoch_disc_loss_list']
        val_recon_epoch_loss_list = checkpoint['val_recon_epoch_loss_list']
        intermediary_images = checkpoint['intermediary_images']
        best_valid_loss = checkpoint['best_valid_loss']

        print(f"Checkpoint loaded. Starting from epoch {start_epoch + 1}")
    else:
        print("No previous checkpoint found. Training from scratch.")
        start_epoch = 0
        epoch_recon_loss_list = []
        epoch_gen_loss_list = []
        epoch_disc_loss_list = []
        val_recon_epoch_loss_list = []
        intermediary_images = []
        best_valid_loss = float("inf")

    return (
        start_epoch,
        epoch_recon_loss_list,
        epoch_gen_loss_list,
        epoch_disc_loss_list,
        val_recon_epoch_loss_list,
        intermediary_images,
        best_valid_loss,
    )


def load_checkpoint_ddpm(model_checkpoint_dir, ddpm, optimizer_diff):
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
        start_epoch = 0
        epoch_loss_list = []
        val_epoch_loss_list = []
        best_valid_loss = float("inf")

    return start_epoch, epoch_loss_list, val_epoch_loss_list, best_valid_loss


def create_transforms():
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.EnsureTyped(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 2)),
            transforms.Resized(keys=["image", "label"], spatial_size=image_size),
            transforms.NormalizeIntensityd(keys=["image"]),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )


def create_dataset(data_dir, section, transforms, num_workers=0, seed=0):
    return DecathlonDataset(
        root_dir=data_dir,
        task="Task01_BrainTumour",
        section=section,
        num_workers=num_workers,
        download=False,
        seed=seed,
        transform=transforms,
        val_frac=0.99 if section == "training" else 0.01,
    )


def create_img_label_list(dataset, mri_channel, label_channel):
    img_list = [dataset[i]["image"][mri_channel, :, :, :] for i in range(len(dataset))]
    label_list = [
        dataset[i]["label"][label_channel, :, :, :] for i in range(len(dataset))
    ]
    return img_list, label_list


def create_combined_tensor(img_list, label_list):
    img_tensor = torch.stack(img_list).unsqueeze(1)
    label_tensor = torch.stack(label_list).unsqueeze(1)
    return torch.cat((img_tensor, label_tensor), dim=1)


def create_data_loader(data, batch_size, shuffle=False, num_workers=0):
    ds = Dataset(data=data)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


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


def train_vqvae(
    train_loader,
    valid_loader,
    vqvae,
    discriminator,
    perceptual_loss,
    adv_loss,
    perceptual_weight,
    adv_weight,
    optimizer_g,
    optimizer_d,
    device,
    start_epoch,
    epochs,
    valid_interval,
    num_example_images,
    original_shape,
    figure_dir,
    plot=True,
):
    total_start = time.time()
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []
    best_valid_loss = float("inf")
    l1_loss = torch.nn.L1Loss()

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

            if images.shape != original_shape:
                # Incompatible image, skipping
                continue

            optimizer_g.zero_grad(set_to_none=True)
            encoded = vqvae.encode_stage_2_inputs(images)
            reconstruction, quantization_loss = vqvae(images)
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            recons_loss = l1_loss(reconstruction.float(), images.float())
            recon_p = reconstruction[:, 0, :, :, :].unsqueeze(1)
            images_p = images[:, 0, :, :, :].unsqueeze(1)
            p_loss = perceptual_loss(recon_p.float(), images_p.float())
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = recons_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

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
                        continue

                    reconstruction, quantization_loss = vqvae(images)

                    if val_step == 1:
                        intermediary_images.append(reconstruction[:num_example_images, 0])
                        if plot:
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
                            plt.savefig(os.path.join(figure_dir, f"reconstruction_{epoch}.png"))


                    recons_loss = l1_loss(reconstruction.float(), images.float())
                    val_loss += recons_loss.item()

                    del images, batch, reconstruction
                    torch.cuda.empty_cache()

            val_loss /= val_step
            val_recon_epoch_loss_list.append(val_loss)

            if plot:
                # Plot losses: reconstruction train and validation
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_recon_loss_list, label="train")
                plt.plot(np.arange(0, len(epoch_recon_loss_list), valid_interval), val_recon_epoch_loss_list, label="validation")
                plt.legend()
                plt.title("Reconstruction loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(os.path.join(figure_dir, "reconstruction_loss.pdf"), format="pdf", bbox_inches="tight")

                # Plot losses : generator and discriminator
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_gen_loss_list, label="generator")
                plt.plot(epoch_disc_loss_list, label="discriminator")
                plt.legend()
                plt.title("Generator and discriminator loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(os.path.join(figure_dir, "generator_discriminator_loss.pdf"), format="pdf", bbox_inches="tight")

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

    total_time = time.time() - total_start
    print(f"Training completed in {total_time // 3600} hours, {(total_time % 3600) // 60} minutes and {total_time % 60} seconds")
    torch.cuda.empty_cache()
    vqvae.eval()

    if plot:
        with torch.no_grad():
            image = first(valid_loader)
            image = image.to(device)
            latent_img = vqvae.encode_stage_2_inputs(image)
            reconstruction, quantization_loss = vqvae(images=image)
        # Plot example latent space on all three axis
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
        plt.savefig(os.path.join(figure_dir, "latent_space_example.pdf"), format="pdf", bbox_inches="tight")

        fig, ax = plt.subplots(4, num_example_images, figsize=(20, 10))
        # Set dpi 
        fig.set_dpi(1000)
        fig.suptitle("Example of VQ-GAN reconstruction", fontsize=16)
        for i in range(num_example_images):
            ax[0, i].set_title(f"Slice : {x_lin[i]+1}", fontsize=12)
            ax[0, i].imshow(image[0, 0, z_lin[i], :, :].cpu().numpy(), cmap="gray")
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
            ax[1, i].imshow(reconstruction[0, 0, z_lin[i], :, :].cpu().numpy())
            ax[1, i].set_xticks([])
            ax[1, i].set_yticks([])
            ax[2, i].imshow(image[0, 1, z_lin[i], :, :].cpu().numpy(), cmap="gray")
            ax[2, i].set_xticks([])
            ax[2, i].set_yticks([])
            ax[3, i].imshow(reconstruction[0, 1, z_lin[i], :, :].cpu().numpy(), cmap="gray")
            ax[3, i].set_xticks([])
            ax[3, i].set_yticks([])
        # Set row titles
        ax[0, 0].set_ylabel("Input", fontsize=12)
        ax[2, 0].set_ylabel("Reconstruction", fontsize=12)
        # Limit space between plots
        plt.subplots_adjust(wspace=0.2, hspace=-0.6)
        # Limit space between plots and suptitle
        fig.subplots_adjust(top=1.2)
        # Save figure
        plt.savefig(os.path.join(figure_dir, "reconstruction_example.pdf"), format="pdf", bbox_inches="tight")

    return (
        epoch_recon_loss_list,
        epoch_gen_loss_list,
        epoch_disc_loss_list,
        val_recon_epoch_loss_list,
        intermediary_images,
        best_valid_loss,
    )


def train_ddpm(
    train_loader,
    valid_loader,
    vqvae,
    ddpm,
    ldm_inferer,
    ddpm_inferer,
    optimizer_diff,
    scaler,
    device,
    start_epoch,
    epochs,
    valid_interval,
    num_example_images,
    original_shape,
    figure_dir,
    plot=True,
):
    epoch_loss_list = []
    val_epoch_loss_list = []
    best_valid_loss = float("inf")

    for epoch in range(start_epoch + 1, epochs):
        ddpm.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:
            images = batch.to(device)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                z = vqvae.encode_stage_2_inputs(images)
                noise = torch.randn_like(z).to(device)
                noise, _ = vqvae.quantize(noise)

                if images.shape != original_shape:
                    continue

                timesteps = torch.randint(
                    0, ldm_inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                noise_pred = ldm_inferer(
                    inputs=images, autoencoder_model=vqvae, diffusion_model=ddpm, noise=noise, timesteps=timesteps
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            del images, noise, noise_pred, loss, timesteps, batch
            torch.cuda.empty_cache()

        epoch_loss_list.append(epoch_loss / (step + 1))

        if (epoch + 1) % valid_interval == 0:
            ddpm.eval()
            val_loss = 0

            with torch.no_grad():
                for step, batch in enumerate(valid_loader):
                    images = batch.to(device)

                    with autocast(enabled=True):
                        z = vqvae.encode_stage_2_inputs(images)
                        noise = torch.randn_like(z).to(device)
                        noise, _ = vqvae.quantize(noise)

                        if images.shape != original_shape:
                            continue

                        timesteps = torch.randint(
                            0, ldm_inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        noise_pred = ldm_inferer(
                            inputs=images, autoencoder_model=vqvae, diffusion_model=ddpm, noise=noise, timesteps=timesteps
                        )

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_loss += loss.item()

                    if step == 1:
                        with torch.no_grad():
                            latent_img = vqvae.encode_stage_2_inputs(images)
                            noise = torch.randn_like(latent_img).to(device)
                            noise, _ = vqvae.quantize(noise)
                            denoised_image = ddpm_inferer.sample(
                                diffusion_model=ddpm,
                                input_noise=noise,
                                scheduler=ldm_inferer.scheduler,
                            )
                            out_image = vqvae.decode_stage_2_outputs(denoised_image)

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
                                ax[0, i].set_xticks([])
                                ax[0, i].set_yticks([])
                                ax[0, i].set_title(f"Slice : {z_lin_pixel[i]+1}", fontsize=12)
                                ax[1, i].imshow(latent_img[0, 0, z_lin_latent[i], :, :].cpu().numpy(), cmap="gray")
                                ax[1, i].set_xticks([])
                                ax[1, i].set_yticks([])
                                ax[1, i].set_title(f"Slice : {z_lin_latent[i]+1}", fontsize=12)
                                ax[2, i].imshow(noise[0, 0, z_lin_latent[i], :, :].cpu().numpy(), cmap="gray")
                                ax[2, i].set_xticks([])
                                ax[2, i].set_yticks([])
                                ax[3, i].imshow(denoised_image[0, 0, z_lin_latent[i], :, :].cpu().numpy(), cmap="gray")
                                ax[3, i].set_xticks([])
                                ax[3, i].set_yticks([])
                                ax[4, i].imshow(out_image[0, 0, z_lin_pixel[i], :, :].cpu().numpy(), cmap="gray")
                                ax[4, i].set_xticks([])
                                ax[4, i].set_yticks([])
                                ax[4, i].set_title(f"Slice : {z_lin_pixel[i]+1}", fontsize=12)
                                ax[5, i].imshow(out_image[0, 1, z_lin_pixel[i], :, :].cpu().numpy())
                                ax[5, i].set_xticks([])
                                ax[5, i].set_yticks([])
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
                            plt.savefig(os.path.join(figure_dir, f"LDM_process_example_{epoch}.pdf"), format="pdf", bbox_inches="tight")


                val_epoch_loss = val_loss / (step + 1)
                val_epoch_loss_list.append(val_epoch_loss)

                if val_epoch_loss < best_valid_loss:
                    best_valid_loss = val_epoch_loss
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

                del images, noise, noise_pred, loss
                torch.cuda.empty_cache()

            if plot:
                # Make loss curves plot
                plt.figure(figsize=(10, 5))
                plt.plot(epoch_loss_list, label="train")
                plt.plot(np.arange(0, epoch + 1, valid_interval), val_epoch_loss_list, label="validation")
                plt.legend()
                plt.title("Denoising (DDPM) Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.savefig(os.path.join(figure_dir, "loss_curves_ddpm.pdf"), format="pdf", bbox_inches="tight")

    return epoch_loss_list, val_epoch_loss_list, best_valid_loss


###################################################################################################
# Settings
###################################################################################################
# Set the multiprocessing method to use the file system (avoids issues with fork during training)
torch.multiprocessing.set_sharing_strategy('file_system')

# Set GPU device
device, device_ids = set_device(verbose=True)

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
image_size = (240, 240, 120)
## Volume size
voxel_size = (2, 2, 2)
## Number of example images
num_example_images = 7
## Latent channels
latent_channels = 8 # 256 # Try 512/64 32/512
## Codebook size
num_embeddings = 1024*4 # 128
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
    loading_transforms = create_transforms()

    # Create training dataset and loader
    train_dataset = create_dataset(data_dir, "training", loading_transforms, seed=seed)
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
    valid_dataset = create_dataset(data_dir, "validation", loading_transforms, seed=seed)
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
    start_epoch, epoch_recon_loss_list, epoch_gen_loss_list, epoch_disc_loss_list, \
        val_recon_epoch_loss_list, intermediary_images, \
        best_valid_loss = load_checkpoint_vqvae(
        vqgan_checkpoint_path,
        vqvae,
        discriminator,
        perceptual_loss,
        optimizer_g,
        optimizer_d
        )

    ###################################################################################################
    # Training : VQ-GAN
    ###################################################################################################

    train_vqvae(
        train_loader=train_loader,
        valid_loader=valid_loader,
        vqvae=vqvae,
        discriminator=discriminator,
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
        in_channels=latent_channels,
        out_channels=latent_channels,

        num_res_blocks=[latent_channels],
        num_channels=[latent_channels],
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
    optimizer_diff = torch.optim.Adam(params=ddpm.parameters(), lr=1e-4)

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
    z = vqvae.encode_stage_2_inputs(util_image)

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
        plot=True,
    )

if __name__ == "__main__":
    main()
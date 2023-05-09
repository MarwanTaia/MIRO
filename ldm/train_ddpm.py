###################################################################################################
# Imports
###################################################################################################
# Local

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
from torch.cuda.amp import autocast


###################################################################################################
# Functions
###################################################################################################
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
# Training
###################################################################################################
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
    ddpm_checkpoint_path,
    plot=True,
):
    epoch_loss_list = []
    val_epoch_loss_list = []
    best_valid_loss = float("inf")
    total_start_time = time.time()

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
                            if plot:
                                ldm_sampling_plot(
                                    figure_dir=figure_dir,
                                    figure_name=f"ldm_sampling_{epoch}",
                                    images=images,
                                    noise=noise,
                                    latent_image=latent_img,
                                    denoised_image=denoised_image,
                                    out_image=out_image,
                                    num_example_images=num_example_images,
                                )

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
                ddpm_loss_plot(
                    figure_dir=figure_dir,
                    figure_name=f"ddpm_loss",
                    epoch_loss_list=epoch_loss_list,
                    val_epoch_loss_list=val_epoch_loss_list,
                    valid_interval=valid_interval,
                    epoch=epoch,
                )

    total_time = time.time() - total_start_time
    print(f"Training completed in {total_time // 3600} hours, {(total_time % 3600) // 60} minutes and {total_time % 60} seconds")

    return epoch_loss_list, val_epoch_loss_list, best_valid_loss
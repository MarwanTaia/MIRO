###################################################################################################
# Imports
###################################################################################################
# General
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from pynvml import *
# PyTorch
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
# MONAI
import monai
from monai.utils import first


def train_vqgan(
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
    vqgan_checkpoint_path,
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
###################################################################################################
# Imports
###################################################################################################
# Local

# General
import os
from pynvml import *
# PyTorch
import torch
# MONAI
import monai
from monai import transforms
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset

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


def create_transforms(image_size):
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
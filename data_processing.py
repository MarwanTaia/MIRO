###################################################################################################
# Imports
###################################################################################################
import os
import shutil
import pydicom
import pandas as pd
import numpy as np
import SimpleITK as sitk
import argparse
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process, Manager, cpu_count
import time
import queue
from PIL import Image, ImageDraw
import pydicom
import skimage

###################################################################################################
# Functions
###################################################################################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_type", choices=["nii", "dcm", "full"], help="type of task to perform")
    parser.add_argument("root_dir", type=str, help="root directory to search for DICOM files")
    parser.add_argument("output_dir", type=str, help="output directory for the cleaned data")

    parser.add_argument("--min_n_imgs", type=int, default=0, help="minimum number of images for a patient")

    parser.add_argument("--max_x_voxel", type=float, default=np.inf, help="maximum x voxel size")
    parser.add_argument("--max_y_voxel", type=float, default=np.inf, help="maximum y voxel size")
    parser.add_argument("--max_z_voxel", type=float, default=np.inf, help="maximum z voxel size")

    parser.add_argument("--desired_x_voxel", type=float, help="desired x voxel size")
    parser.add_argument("--desired_y_voxel", type=float, help="desired y voxel size")
    parser.add_argument("--desired_z_voxel", type=float, help="desired z voxel size")

    parser.add_argument("--n_workers", type=int, default=1, help="number of workers to use")
    parser.add_argument("--plot", action="store_true", help="plot the original and resampled images")

    return parser.parse_args()


def setup_directories(args):
    os.makedirs(args.output_dir, exist_ok=True)
    return pd.DataFrame(columns=['dp', 'x_voxel', 'y_voxel', 'z_voxel', 'x_resolution', 'y_resolution', 'z_resolution'])


def resample_image(image, original_spacing, new_spacing):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(tuple(int(np.round(image.GetSize()[i] * original_spacing[i] / new_spacing[i])) for i in range(3)))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputPixelType(image.GetPixelID())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


def check_dicom_files(dirpath):
    return any(fname.endswith('.dcm') for fname in os.listdir(dirpath))


def create_output_subdir(args, pt_num):
    pt = f"pt_{pt_num:03d}"
    output_subdir = os.path.join(args.output_dir, pt)
    os.makedirs(output_subdir, exist_ok=True)
    return pt, output_subdir


def copy_dicom_files(dirpath, output_subdir):
    for fname in os.listdir(dirpath):
        if fname.endswith('.dcm'):
            input_path = os.path.join(dirpath, fname)
            output_path = os.path.join(output_subdir, fname)
            shutil.copyfile(input_path, output_path)


def check_voxel_size(args, voxel_size):
    return (voxel_size[0] > args.max_x_voxel) or (voxel_size[1] > args.max_y_voxel) or (voxel_size[2] > args.max_z_voxel)


def save_sitk_image_to_nifti(sitk_image, output_path, original_size=None, patient=None):
    sitk_image = sitk.PermuteAxes(sitk_image, [1, 2, 0])
    print(f"Shape of sitk_image: {sitk_image.GetSize()}")
    print(f"Shape of resampled image: {patient.get_resampled_resolution()}")
    print(f"From {patient.get_default_resolution()} to {patient.get_resampled_resolution()}")

    # Center the image if needed
    if original_size is not None:
        centered_image = center_image(sitk_image, original_size)
        sitk.WriteImage(centered_image, output_path)
    else:
        sitk.WriteImage(sitk_image, output_path)


def create_new_row(dp, voxel_size, resolution):
    return pd.DataFrame({'dp': [dp],
                         'x_voxel': [voxel_size[0]],
                         'y_voxel': [voxel_size[1]],
                         'z_voxel': [voxel_size[2]],
                         'x_resolution': [resolution[0]],
                         'y_resolution': [resolution[1]],
                         'z_resolution': [resolution[2]]})


def plot_images(args, output_path, dp):
    original_image = sitk.ReadImage(output_path)
    original_image_array = sitk.GetArrayFromImage(original_image)
    resampled_image = sitk.ReadImage(output_path)
    resampled_image_array = sitk.GetArrayFromImage(resampled_image)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_array[original_image_array.shape[0] // 2, :, :], cmap='gray')
    plt.title('Original image')
    plt.axis('off')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.imshow(resampled_image_array[resampled_image_array.shape[0] // 2, :, :], cmap='gray')
    plt.title('Resampled image')
    plt.axis('off')
    plt.axis('equal')

    plt.savefig(os.path.join(args.output_dir, f"{dp}.png"))
    plt.close()
    plt.clf()

def make_patient(dirpath, dp):
    # We process the files if they are DICOM files and have "CT" in their name
    # files = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath) if fname.endswith('.dcm') and ("CT" in fname or "RS" in fname)]
    files = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath) if fname.endswith('.dcm')] # and ("CT" in fname or "RS" in fname)]

    # If there are no DICOM files, skip the directory
    if len(files) == 0:
        print(f'No DICOM files found in {dirpath}')
        return None
    
    # Collect the RT file, if it exists and exclude it from the list of files
    rt_struct = None
    for f in files:
        if "RS" in f:
            rt_struct = f
            files.remove(f)
            break
    
    # Sort the files by their slice location
    if pydicom.dcmread(files[0]).get('SliceLocation') is None:
        print(f'No SliceLocation found in {dirpath}')
        return None
    sorted_slices = sorted([(f, pydicom.dcmread(f).SliceLocation) for f in files], key=lambda x: x[1])
    files = [f[0] for f in sorted_slices]

    # Initialize the patient object
    patient = Patient(files, rt_struct)

    return patient


def save_results(args, results_df, rows):
    for row in rows:
        if row is not None:
            results_df = pd.concat([results_df, row], ignore_index=True)

    output_csv = os.path.join(args.output_dir, 'data_index.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}.")


def worker_process_directory(dir_queue, results_list, args):
    while not dir_queue.empty():
        try:
            dirpath, dp_num = dir_queue.get(timeout=1)
            row = process_directory(dirpath, dp_num, args)
            if row is not None:
                results_list.append(row)
        except queue.Empty:
            break


def explore_subdirectories(dirpath):
    subdirs = [os.path.join(dirpath, d) for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))]
    for subdir in subdirs:
        if check_dicom_files(subdir):
            yield subdir
        else:
            yield from explore_subdirectories(subdir)


def center_image(sitk_image, original_size):
    new_size = sitk_image.GetSize()
    diff = [new_size[i] - original_size[i] for i in range(3)]
    padding = [(-diff[i] // 2, -diff[i] // 2) for i in range(3)]
    
    centered_image = sitk.Pad(sitk_image, padding)
    return centered_image


###################################################################################################
# Classes
###################################################################################################

class Patient:
    def __init__(self, ct_files, rt_file):
        self.ct_files = self.load_ct_files(ct_files)
        self.rt_file = rt_file
        # None
        self.resampled_array = None

    def load_ct_files(self, ct_files):
        ct_files = [pydicom.dcmread(file) for file in ct_files]
        ct_files = sorted(ct_files, key=lambda x: x.ImagePositionPatient[2])
        return ct_files
    
    def make_sitk_image(self, img_array=None):
        ct_array = self.get_ct_array() if img_array is None else img_array
        ct_image = sitk.GetImageFromArray(ct_array)
        ct_image = sitk.PermuteAxes(ct_image, [1, 2, 0])
        ct_image.SetSpacing(self.get_voxel_size())
        return ct_image
    
    def resample_image(self, desired_voxel_size, img_array=None):
        ct_image = self.make_sitk_image(img_array)
        original_voxel_size = self.get_voxel_size()
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(desired_voxel_size)
        # Keep original dimensions
        # resample_filter.SetSize(ct_image.GetSize())
        resample_filter.SetSize([int(round(ct_image.GetSize()[i] * original_voxel_size[i] / desired_voxel_size[i])) for i in range(3)])
        resample_filter.SetOutputDirection(ct_image.GetDirection())
        resample_filter.SetOutputOrigin(ct_image.GetOrigin())
        resample_filter.SetOutputPixelType(ct_image.GetPixelID())
        resample_filter.SetTransform(sitk.Transform())
        # resample_filter.SetDefaultPixelValue(-1024)
        resample_filter.SetInterpolator(sitk.sitkLinear)
        resampled_image = resample_filter.Execute(ct_image)
        resampled_image = sitk.PermuteAxes(resampled_image, [2, 0, 1])
        resampled_array = sitk.GetArrayFromImage(resampled_image)
        resampled_array = resampled_array.astype(np.int16)
        # resampled_array = resampled_array * self.ct_files[0].RescaleSlope + self.ct_files[0].RescaleIntercept
        self.resampled_array = resampled_array
        return resampled_array, resampled_image
    
    def make_mask(self, roi_name='OralCavity'):
        if self.rt_file is None:
            return None
        rt_file = pydicom.dcmread(self.rt_file)
        roi_number = None
        for roi in rt_file.StructureSetROISequence:
            if roi.ROIName == roi_name:
                roi_number = roi.ROINumber
                break
        if roi_number is None:
            return None
        roi_contour = None
        for contour in rt_file.ROIContourSequence:
            if contour.ReferencedROINumber == roi_number:
                roi_contour = contour
                break
        if roi_contour is None:
            return None
        mask = np.zeros(self.get_default_resolution(), dtype=np.uint8)
        for contour in roi_contour.ContourSequence:
            slice_dict = {}
            
            # List of DICOM coordinates
            slice_dict['XY_dcm'] = list(zip(np.array(contour.ContourData[0::3]), np.array(contour.ContourData[1::3])))
            slice_dict['Z_dcm'] = float(contour.ContourData[2])

            # List of coordinates in the image frame
            slice_dict['XY_img'] = list(zip(((np.array(contour.ContourData[0::3]) - self.ct_files[0].ImagePositionPatient[0]) / self.ct_files[0].PixelSpacing[0]).astype(np.int16),
                                        ((np.array(contour.ContourData[1::3]) - self.ct_files[0].ImagePositionPatient[1]) / self.ct_files[0].PixelSpacing[1]).astype(np.int16)))
            slice_dict['Z_img'] = int((slice_dict['Z_dcm'] - self.ct_files[0].ImagePositionPatient[2]) / self.ct_files[0].SliceThickness)
            slice_dict['slice_id'] = int(round(slice_dict['Z_img']))

            # Convert polygon to mask (based on PIL, fast)
            img = Image.new('L', (self.ct_files[0].Rows, self.ct_files[0].Columns), 0)
            ImageDraw.Draw(img).polygon(slice_dict['XY_img'], outline=1, fill=1)
            mask[:, :, slice_dict['slice_id']] = np.logical_or(mask[:, :, slice_dict['slice_id']], np.array(img))
        return mask
    
    def get_ct_array(self):
        ct_array = np.stack([file.pixel_array for file in self.ct_files])
        ct_array = ct_array.astype(np.int16)
        # ct_array = ct_array * self.ct_files[0].RescaleSlope + self.ct_files[0].RescaleIntercept
        # adjust HU values for each slice
        for i in range(ct_array.shape[0]):
            ct_array[i] = ct_array[i] * self.ct_files[i].RescaleSlope + self.ct_files[i].RescaleIntercept
        # permute [1, 2, 0]
        ct_array = np.transpose(ct_array, (1, 2, 0))
        return ct_array
    
    def get_voxel_size(self):
        try:
            return self.ct_files[0].PixelSpacing[0], self.ct_files[0].PixelSpacing[1], self.ct_files[0].SliceThickness
        except:
            return None, None, None
    
    def get_default_resolution(self):
        try:
            return self.ct_files[0].Rows, self.ct_files[0].Columns, len(self.ct_files)
        except:
            return None, None, None
    
    def get_resampled_resolution(self):
        try:
            return self.resampled_array.shape
        except:
            return None, None, None
        
    def get_hounsfield_range(self):
        try:
            return self.ct_files[0].RescaleSlope, self.ct_files[0].RescaleIntercept
        except:
            return None, None

###################################################################################################
# Main
###################################################################################################


def process_directory(dirpath, dp_num, args):
    if not check_dicom_files(dirpath):
        return

    dp, output_subdir = create_output_subdir(args, dp_num) if args.task_type == "full" else (f"dp_{dp_num:03d}", None)

    patient = make_patient(dirpath, dp)

    if patient is None:
        print(f"Skipping {dirpath} because it does not contain any DICOM files.")
        return
    
    # Check if the patient has enough slices
    if len(patient.ct_files) < args.min_n_imgs:
        return
    
    # Check if the patient has the maximum number of voxels
    voxel_size = patient.get_voxel_size()
    if check_voxel_size(args, voxel_size):
        return

    if args.task_type == "full":
        copy_dicom_files(dirpath, output_subdir)

    desired_voxel_size = (args.desired_x_voxel, args.desired_y_voxel, args.desired_z_voxel) if args.desired_x_voxel and args.desired_y_voxel and args.desired_z_voxel else patient.get_voxel_size()

    volume, sitk_image = patient.resample_image(desired_voxel_size)

    # mask = patient.make_mask('SpinalCord')
    mask = None

    # make mask a second channel of the volume
    if mask is not None:
        mask, _ = patient.resample_image(desired_voxel_size, mask)
        # print(f"mask and volume shapes: {mask.shape}, {volume.shape}")
        # add a channel dimension
        mask = np.expand_dims(mask, axis=0)
        volume = np.expand_dims(volume, axis=0)
        volume = np.concatenate((volume, mask), axis=0)

        # fig, ax = plt.subplots(2, 10, figsize=(20, 4))
        # for i in range(10):
        #     ax[0, i].imshow(volume[0, :, :, i*10], cmap="gray")
        #     ax[1, i].imshow(volume[1, :, :, i*10], cmap="gray")
        #     ax[0, i].axis('off')
        #     ax[1, i].axis('off')
        #     ax[0, i].set_title(f"Slice {i*10}")
        # # set row titles
        # ax[0, 0].set_ylabel("CT")
        # ax[1, 0].set_ylabel("Mask")
        # plt.tight_layout()
        # plt.show()
    else:
        print(f"No mask found for {dp}.")

    output_path = os.path.join(output_subdir, f"{dp}.nii.gz") if args.task_type == "full" else os.path.join(args.output_dir, f"{dp}.nii.gz")
    save_sitk_image_to_nifti(sitk_image, output_path, patient=patient)

    new_row = create_new_row(dp, voxel_size, patient.get_resampled_resolution())

    if args.plot:
        plot_images(args, output_path, dp)

    return new_row

def main():
    args = parse_arguments()
    print("Got arguments.")
    results_df = setup_directories(args)
    print("Created output directory.")

    start_time = time.time()
    dirs_to_process = []
    print("Searching for DICOM files...")
    for dirpath in explore_subdirectories(args.root_dir):
        dirs_to_process.append(dirpath)
    print(f"Found {len(dirs_to_process)} directories to process.")

    dir_queue = Manager().Queue()
    for i, dirpath in enumerate(dirs_to_process):
        dir_queue.put((dirpath, i))

    results_list = Manager().list()

    print("Processing directories...")
    processes = []
    n_workers = args.n_workers if args.n_workers > 0 else cpu_count()
    for _ in range(n_workers):
        p = Process(target=worker_process_directory, args=(dir_queue, results_list, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Processed {len(results_list)} directories in {time.time() - start_time:.2f} seconds.")
    print("Saving results to CSV file...")
    save_results(args, results_df, results_list)

if __name__ == "__main__":
    main()
    print("Done.")
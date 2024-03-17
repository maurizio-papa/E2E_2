import numpy as np
import h5py
from PIL import Image
import io
import os 

def images_to_hdf5(input_directory, output_file):
    '''
    Convert images in input_directory to HDF5 format.
    '''
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]
    
    with h5py.File(output_file, 'w') as hf:
        for image_file in image_files:
            image_path = os.path.join(input_directory, image_file)
            with open(image_path, 'rb') as img_f:
                binary_data = img_f.read()
            binary_data_np = np.asarray(binary_data)
            hf.create_dataset(image_file, data=binary_data_np)


def load_images_from_hdf5(input_hdf5_file):
    '''
    Load images from HDF5 file.
    Args:
        input_hdf5_file (str): Input HDF5 file containing images.
    Returns:
        images_dict (dict): Dictionary containing image names as keys and corresponding NumPy arrays as values.
    '''
    images_dict = {}
    with h5py.File(input_hdf5_file, 'r') as hf:
        for key in hf.keys():
            images_dict[key] = Image.open(io.BytesIO(np.array(hf[key])))
    return images_dict


def batch_images_to_hdf5(input_directory, output_directory, output_prefix, batch_size=10, stride=5):
    '''
    Convert images in input_directory to HDF5 format with specified batch size and stride.
    Create separate HDF5 files for each batch.
    '''
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg')]
    total_images = len(image_files)
    num_batches = (total_images - batch_size) // stride + 1

    for i in range(num_batches):
        batch_images = image_files[i * stride: i * stride + batch_size]
        print(batch_images)

        output_file = f"{output_directory}/{output_prefix}_batch_{i + 1}.h5"
        with h5py.File(output_file, 'w') as hf:
            for j, image_file in enumerate(batch_images):
                image_path = os.path.join(input_directory, image_file)
                with open(image_path, 'rb') as img_f:
                    binary_data = img_f.read()
                binary_data_np = np.asarray(binary_data)
                hf.create_dataset(f'image_{j + 1}', data=binary_data_np)

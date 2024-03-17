import os 
from image_to_tensor_h5 import load_images_from_hdf5, batch_images_to_hdf5
import tarfile
import logging 


def extract_tar(tar_file, destination_folder):
    try:
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(destination_folder)
        print(f"Extraction complete. Files extracted to: {destination_folder}")
        os.remove(tar_file)
    except Exception as e:
        print(f"Error extracting the tar file: {e}")

def untar_directories(source_folder, destination_folder):
    for directory in os.listdir(source_folder):
        for file in os.listdir(f'{source_folder}/{directory}'):
            file_path = f'{source_folder}/{directory}/{file}'
            dst = f'{destination_folder}/{directory}/{file.split(".")[0]}'
            if not os.path.exists(dst):
                os.makedirs(dst)
            try:
                extract_tar(file_path, dst)
            except Exception as e:
                print(f"Failed to untar {file_path}. Error: {str(e)}")


def main():
    source = 'images'
    destination = 'untared_images'
    untar_directories(source, destination)
    for directory in os.listdir(destination):
        for dir in os.listdir(f'{destination}/{directory}'):
            if not os.path.exists(f'tensor/{directory}/{dir}'):
                os.makedirs(f'tensor/{directory}/{dir}')
            batch_images_to_hdf5(f'{destination}/{directory}/{dir}', 
                                 f'tensor/{directory}/{dir}', 'hdf_img', batch_size=50, stride= 25)


if __name__ == '__main__':
    main()





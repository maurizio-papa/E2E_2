from .rgb_to_tensor.image_to_tensor_h5 import load_images_from_hdf5, batch_images_to_hdf5

import tarfile
import os 

def extract_tar(tar_file, destination_folder):
    try:
        with tarfile.open(tar_file, 'r') as tar:
            tar.extractall(destination_folder)
    except Exception as e:
        print(f"Error {e}")



def untar_directories(img_dirs, destination_dir):
    for directory in os.listdir(img_dirs):
        for idx, file in enumerate(os.listdir(os.path.join(img_dirs, directory))):
            sub_dst_dir = os.path.join(destination_dir, os.path.join(directory, f'{idx}'))
            if not os.path.exists(sub_dst_dir):
                os.makedirs(sub_dst_dir)
            source = os.path.join(os.path.join(img_dirs, directory), file)
            extract_tar(source, sub_dst_dir)



def main():
    untar_directories('images', 'untared_images')

    print('finished untaring directories')

    for dir in os.listdir('untared_images'):
        if not os.path.exists(f'tensor/{dir}'):
            os.makedirs(f'tensor/{dir}')
        for sub_dir in os.listdir(os.path.join('untared_images', dir)):
            dst_output = f'tensor/{dir}/{sub_dir}'
            if not os.path.exists(dst_output):
                os.makedirs(dst_output)
            print(f'processing following dir: {dir}')
            batch_images_to_hdf5(f'untared_images/{dir}/{sub_dir}',dst_output, 'hdf_img', batch_size=50, stride= 25)


if __name__ == '__main__':
    main()
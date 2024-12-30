### Dependencies

# Utils
import numpy as np
import os
from PIL import Image


### Image methods

def load_images_f(data_folder: str, ext: str = '.raw'):
    """
    Loads the images.
    """

    # Check if folder exists
    if not os.path.isdir(data_folder):
        raise OSError(f'{data_folder}, directory does not exist.')
    
    # Set holding lists, lists to hold tuples
    inputs, _, targets = 'Baseline', '', 'Target'
    input_images, _, target_images = [], [], []

    # Iterate through the directory
    for folder_path, _, images in os.walk(data_folder):
        for img in images:
            file_path = os.path.join(folder_path, str(img))
            if os.path.isfile(file_path) and img.endswith(ext):
                try:
                    if inputs in str(file_path):
                        load_image_logic_f(file_path, str(img), input_images)
                    elif targets in str(file_path):
                        load_image_logic_f(file_path, str(img), target_images)
                    
                except Exception as e:
                    print(f'Error loading {img}: {e}')
    
    return input_images, _, target_images

def load_image_logic_f(file_path, file_name: str, data_list: list, flag: bool= False):
    """
    More logic on loading different images in the dataset
    """

    # Loads file from filepath into array
    file = np.fromfile(file_path, dtype='float32', sep="")

    # # Sets excessive values to floor value (purpose: zeros metal object)
    # file[file > 2999.9] = -500

    # # Clips values to specified min-max range
    # file = np.clip(file, -500, 2999)

    # # Normalizes array values to 0 - 1 range
    # file = (file - (-500)) / (2999 - (-500))

    # # Reshape array, expecting array ([512**2])
    # file = file.reshape([512, 512])
    # file_name = file_name[file_name.find('img'):file_name.find('x1', file_name.find('img'))]
    data_list.append(file)

    return

def mask_images(inputs, targets, threshold):
    """
    Masks metal from images given a threshold, based on HU
    """

    masked_inputs, masked_labels = [], []

    # Iterate through dataset
    for input, target in zip(inputs, targets):

        # Define metal mask from input
        mask_indices = np.nonzero(input > threshold)
        
        # Mask metal in both input and target
        input[mask_indices] = -500
        target[mask_indices] = -500

        # Clip excessive ranges
        input = np.clip(input, -500, threshold)
        target = np.clip(target, -500, threshold)

        # Forgot this part
        input = (input - (-500)) / (threshold - (-500))
        target = (target - (-500)) / (threshold - (-500))

        # Reshape from original [1, 512, 512] shape
        input = input.reshape([512, 512])
        target = target.reshape([512, 512])

        # Send
        masked_inputs.append(input), masked_labels.append(target)
    
    return masked_inputs, masked_labels


### Sorting methods

def load_image(image_path: str, ext: str = ".raw"):
    image_holder = []
    if os.path.isfile(image_path) and image_path.endswith(ext):
        try:
            image = load_image_logic_simple(image_path)
        except Exception as e:
            print(f'Error loading {image_path}: {e}')
        
    return image

def load_image_logic_simple(image_path: str):
    file = np.fromfile(image_path, dtype='float32', sep="")
    file = np.clip(file, -500, 2999)
    file = (file - (-500)) / (2999 - (-500))
    if file.size == (512**2): # or file.size == (900*1000):
        file = file.reshape([512, 512]) if file.size == (512**2) else file.reshape([900, 1000])
        # file_name = file_name[file_name.find('img'):file_name.find('x1', file_name.find('img'))] # file_name[file_name.find('img'):file_name.find('_', file_name.find('img'))] # file_name[:-4]
        # list_holder.append(file) #(file, file_path[:file_path.rfind('\\')], file_name))
    return file

def load_images(data_folder: str, ext: str = '.raw', flag: bool= False):
    """
    Loads the images.
    """
    # Check if folder exists
    if not os.path.isdir(data_folder):
        raise OSError(f'{data_folder}, directory does not exist.')
    
    # Set holding lists, lists to hold tuples
    inputs, masks, targets = 'Baseline', 'Mask', 'Target'
    input_images, mask_images, target_images = [], [], []

    # Iterate through the directory
    for folder_path, _, images in os.walk(data_folder):
        for img in images:
            file_path = os.path.join(folder_path, str(img))
            if os.path.isfile(file_path) and img.endswith(ext):
                try:
                    if inputs in str(file_path):
                        load_image_logic(file_path, str(img), input_images, flag)
                    elif targets in str(file_path):
                        load_image_logic(file_path, str(img), target_images, flag)
                    # else:
                        # load_image_logic(file_path, str(img), mask_images)
                except Exception as e:
                    print(f'Error loading {img}: {e}')
    
    return input_images, mask_images, target_images

def load_image_logic(file_path, file_name: str, data_list: list, flag: bool= False):
    """
    More logic on loading different images in the dataset
    """

    # Loads file from filepath into array
    file = np.fromfile(file_path, dtype='float32', sep="")

    # Sets excessive values to floor value (purpose: zeros metal object)
    file[file > 2999.9] = -500

    # Clips values to specified min-max range
    file = np.clip(file, -500, 2999)

    # Normalizes array values to 0 - 1 range
    file = (file - (-500)) / (2999 - (-500))

    # Reshape array
    if file.size == (512**2): # or file.size == (900*1000):
        file = file.reshape([512, 512]) if file.size == (512**2) else file.reshape([900, 1000])
        file_name = file_name[file_name.find('img'):file_name.find('x1', file_name.find('img'))] # file_name[file_name.find('img'):file_name.find('_', file_name.find('img'))] # file_name[:-4]
        data_list.append(file) #(file, file_path[:file_path.rfind('\\')], file_name))

    return

def filter_image_pairs(data_folder: str, ext: str= '.raw'):
    """
    Separates input-mask-target pairs from ones without and writes them to a file.
    """
    input_paths, mask_paths, target_paths = {}, {}, {}
    inputs, masks, targets = 'Baseline', 'Mask', 'Target'
    pairable_paths = []
    write_to_file = 'filtered_pairs.txt'

    for folder_path, _, images in os.walk(data_folder):
        for img in images:
            file_name = str(img)
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and img.endswith(ext):
                file_name = file_name[file_name.find('img'):file_name.find('_', file_name.find('img'))]
                if inputs in file_path:
                    input_paths[file_name] = file_path[file_path.find('\\', file_path.find('Diff'))+1:]
                elif masks in file_path:
                    mask_paths[file_name] = file_path[file_path.find('\\', file_path.find('Diff'))+1:]
                else:
                    target_paths[file_name] = file_path[file_path.find('\\', file_path.find('Diff'))+1:]
    
    pairable_images = set(input_paths.keys()) & set(mask_paths.keys()) & set(target_paths.keys())

    for key in pairable_images:
        pairable_paths.append(input_paths[key])
        pairable_paths.append(mask_paths[key])
        pairable_paths.append(target_paths[key])
    
    with open(write_to_file, 'w') as file:
        for path in pairable_paths:
            file.write(path + '\n')

def load_filtered_pairs(file_path: str):
    """
    Loads pairable images from written file
    """
    origin_path = 'C:\\1-\\ct-proc-data\\'
    inputs, masks, targets = 'Baseline', 'Mask', 'Target'
    input_images, mask_images, target_images = [], [], []

    with open(file_path, 'r') as file:
        paths = file.read().splitlines()

        for path in paths:
            if inputs in path:
                load_image_logic(origin_path + path[path.find('body'):], path[path.find('train'):], input_images)
            elif masks in path:
                load_image_logic(origin_path + path[path.find('body'):], path[path.find('train'):], mask_images)
            else:
                load_image_logic(origin_path + path[path.find('body'):], path[path.find('train'):], target_images)
    
    return input_images, mask_images, target_images

def format_image(file):
    '''
    Converts to image, then keeps at 1 channel (formally: adds 2 channels for rgb input and a batch channel)
    '''
    img = Image.fromarray(file, mode= 'F')
    # img = img.convert('RGB')
    return img

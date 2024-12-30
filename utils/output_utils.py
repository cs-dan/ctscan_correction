### Dependencies
import os


### Methods

def save_to_disk(path: str, images, names):
    """ Saves figure to run folder. Must be matplotlib pyplot figure object. """

    folder_path = path

    # Check if folder exists
    os.makedirs(folder_path, exist_ok= True)

    # Read from args and save to disk
    for image, name in zip(images, names):
        file_path = os.path.join(folder_path, name)
        image.savefig(f"{file_path}.png", dpi= 300)

    return

def write_log(path : str, run_info):
    """ Writes run information. """

    # Write to info file
    with open(f'{path}\\info.txt', 'w') as file:

        # Format: Sample Name, Num Iters, Learning Rate, SSD @ End
        file.write(f'sample name: {run_info[0]}\n')
        file.write(f'number of iterations: {run_info[1]}\n')
        file.write(f'learning rate: {run_info[2]}\n')
        file.write(f'final ssd value: {run_info[3]:.4f}\n')

def get_run(path='results'):
    """ Gets the next run number and makes the folder for it. """

    current_run = 1

    # Check if path exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get all existing run folders
    existing_runs = [
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('run_')
    ]

    # Parse to find highest run number
    for run in existing_runs:
        try:
            run_num = int(run.split('_')[1])
            current_run = run_num + 1 if (run_num + 1) > current_run else current_run
        except (IndexError, ValueError):
            pass
    
    run_path = f'Results\\run_{current_run}'

    # Create new run folder
    os.makedirs(run_path)

    return run_path

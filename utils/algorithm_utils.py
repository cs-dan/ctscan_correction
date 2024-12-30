### Dependencies

# System tools
import sys

# Self tools
from model_utils import *
from algorithm_utils import *
from output_utils import *

# Sim tools
from transform_utils import projection, reconstruction

# Model tools
import torch
import torch.nn as nn

# Utils
import numpy as np
import matplotlib.pyplot as pyp
import matplotlib.pylab as plt

# Console capture
import os
import contextlib


### Methods

def frame_logging(*args):
    """ Graph various logs for visualization by iteration. """

    name = f'logs-plots'

    # Arguments
    network_output = args[0]
    loss_by_iter = args[1]
    step_movement = args[2]
    num_iters = range(1, args[3] + 1)

    # Prep
    graphs = [network_output, step_movement]

    # Figure, Axis creation
    fig, (axis_1, axis_2) = pyp.subplots(1, 2, figsize=(24, 8))

    # Ploting using pyplot
    axis_1.plot(num_iters, graphs[0], label= "Output", color= 'teal', linewidth= 2)
    axis_1.set_title("Network Output Over Descent", fontsize= 16, fontweight= "bold")
    axis_1.set_xlabel("Iteration Number", fontsize= 12)
    axis_1.set_ylabel("Output Value", fontsize= 12)
    axis_1.grid(True, which= "both", linestyle= "-", linewidth= 1)
    axis_1.set_facecolor("#f0f0f0")
    axis_1.legend(loc= "best")

    axis_2.plot(num_iters, graphs[1], label= "Step", color= 'green', linewidth= 2)
    axis_2.set_title("SSD by Iteration", fontsize= 16, fontweight= "bold")
    axis_2.set_xlabel("Iteration Number", fontsize= 12)
    axis_2.set_ylabel("SSD Value", fontsize= 12)
    axis_2.grid(True, which= "both", linestyle= "-", linewidth= 0.5)
    axis_2.set_facecolor("#f0f0f0")
    axis_2.legend(loc= "best")

    pyp.tight_layout()

    # Show plot
    pyp.show()

    return fig, name

def frame_sinograms(*args):
    """ Sinogram view plotting function for 2 images. """

    # Load the original, load the fresh output -> already loaded
    input_image = args[0]
    output_image = args[1]

    # Reshape from extra dims
    input_image = input_image.reshape([1000, 900])
    output_image = output_image.reshape([1000, 900])

    # Plot using pyplot
    pyp.figure(figsize= (24, 8))

    for i, image in enumerate([input_image, output_image]):
        pyp.subplot(1, 2, i+1)
        plt.imshow(image, cmap= plt.cm.Greys_r, vmin= 0, vmax= 8)
        pyp.title(f"Image {i+1}")
    
    pyp.show()

def frame_scans(*args):
    """ Image view plotting. """
    
    # Load the original image, load the fresh output image
    input_image = args[0]
    output_image = args[1]
    clean_image = args[2]
    iter_num = args[3]
    model_output = args[4]
    ssd_value = args[5]

    # Clip excessive ranges
    input_image = np.clip(input_image, -500, 2999)
    output_image = np.clip(output_image, -500, 2999)
    clean_image = np.clip(clean_image, -500, 2999)
    
    # Forgot this part
    input_image = (input_image - (-500)) / (2999 - (-500))
    output_image = (output_image - (-500)) / (2999 - (-500))
    clean_image = (clean_image - (-500)) / (2999 - (-500))
    
    # Reshape from original [1, 512, 512] shape
    input_image = input_image.reshape([512, 512])
    output_image = output_image.reshape([512, 512])
    clean_image = clean_image.reshape([512, 512])

    images = [input_image, output_image, clean_image]

    fig = pyp.figure(figsize= (36, 12))
    plt.suptitle(f"Comparison at {iter_num} | Output: {model_output:.4f} | SSD: {ssd_value:.4f}")

    # Plot image using pyplot
    for i, image in enumerate(images):
        pyp.subplot(1, len(images), i+1)
        plt.imshow(image, cmap= plt.cm.Greys_r, vmin= 0, vmax= 1)

    plt.close(fig)

    name = f'comparison_{iter_num}'
    
    return fig, name

def frame_iteration(*args):
    """ Code to visualize an iteration's gradients, etc. """
    
    raw = args[0].squeeze().reshape([512,512])
    grads_orig = args[1].squeeze().reshape([512,512])
    grads = args[2].squeeze().reshape([512, 512])
    grads_sino = args[3].squeeze().reshape([1000, 900])
    sino = args[4].squeeze().reshape([1000, 900])
    iter_num = args[5]
    clean = args[6].squeeze().reshape([512,512])

    raw = np.clip(raw, -500, 2399)
    raw = (raw - (-500)) / (2399 - (-500))

    clean = np.clip(clean, -500, 2399)
    clean = (clean - (-500)) / (2399 - (-500))

    fig = plt.figure(figsize=(18, 9))

    # Clean Version (Goal)
    plt.subplot(2, 3, 1)
    plt.title("Clean Image")
    plt.imshow(clean, cmap= 'grey')

    # Current Modified 
    plt.subplot(2, 3, 2)
    plt.title("Current Image")
    plt.imshow(raw, cmap= 'grey')

    # Classifier Gradient map
    plt.subplot(2, 3, 3)
    plt.title("Input Level Gradient Map")
    plt.imshow(grads_orig, cmap= 'hot')
    plt.colorbar()

    # Scan gradient map
    plt.subplot(2, 3, 4)
    plt.title("Metal Mask Map")
    plt.imshow(grads, cmap= 'hot')
    plt.colorbar()

    # Sinosoid gradient map
    plt.subplot(2, 3, 5)
    plt.title("Sinogram Metal Map")
    plt.imshow(grads_sino, cmap='hot')
    plt.colorbar()

    # Original Sinogram
    plt.subplot(2, 3, 6)
    plt.title("Current Sinogram")
    plt.imshow(sino, cmap='grey')

    plt.suptitle(f"Iteration Step {iter_num}", fontsize=16, fontweight= 'bold')

    pyp.show()

    name = f'view_{iter_num}'

    return fig, name

def ssd(*args):
    """ Sum squared difference comparison of the afflicted and corrected image. """
    
    true_image = args[0] 
    output_image = args[1]
    image_domain = args[2]

    # Run preprocs
    if image_domain == 'scan':
        true_image = np.clip(true_image, -500, 2999)
        output_image = np.clip(output_image, -500, 2999)
        true_image = true_image.reshape([512, 512])
        output_image = output_image.reshape([512, 512])
    else:
        true_image = true_image.reshape([1000, 900])
        output_image = output_image.reshape([1000, 900])        

    # Run squared difference routine
    upper = np.sum((true_image - output_image)**2)
    lower = ((np.sum(true_image**2))*(np.sum(output_image**2)))**0.5
    solve = upper / lower

    # print(f"SSD Results on {image_domain}: {solve:.4f}")

    return solve

def write_raw(filename, image):
    """ Writes raw image to disk. """

    with open(filename, 'wb') as fout:
        fout.write(image)

def prepare_raw(image, ceiling):
    """ Prepares raw for classifier input. """

    # Set value ceiling, preproc image
    image[image > ceiling] = -500
    image = np.clip(image, -500, ceiling)
    image = (image - (-500)) / (ceiling - (-500))
    image = image.reshape([512, 512])

    # Apply transforms and to tensor
    image = transform_config(image)
    image = image.unsqueeze(0)

    image.requires_grad_()

    return image

def prepare_mask(input, threshold):
    """ Defines and creates a mask based on an input. """

    # Boolean mask
    mask = input > threshold

    print(f"Number of selection of mask: {np.sum(mask)}")

    return mask

def frame_scans_single(*args):
    """ Image view plotting. """
    
    # Load the original image, load the fresh output image
    output_image = args[0]
    iter_num = args[1]
    model_output = args[2]

    # Clip excessive ranges
    output_image = np.clip(output_image, -500, 2999)
    
    # Forgot this part
    output_image = (output_image - (-500)) / (2999 - (-500))

    # Reshape from original [1, 512, 512] shape
    output_image = output_image.reshape([512, 512])

    fig = pyp.figure(figsize= (12, 12))
    plt.suptitle(f"Iteration {iter_num} | Output: {model_output:.4f}")

    # Plot image using pyplot
    plt.imshow(output_image, cmap= plt.cm.Greys_r, vmin= 0, vmax= 1)

    plt.close(fig)

    name = f'comparison_{iter_num}'
    
    return fig, name

def frame_logging_single(*args):
    """ Graph various logs for visualization by iteration. """

    name = f'logs-plots'

    # Arguments
    network_output = args[0]
    num_iters = range(1, args[1] + 1)

    # Prep
    graphs = [network_output]

    # Figure, Axis creation
    # fig, (axis_1, axis_2) = pyp.subplots(1, 2, figsize=(24, 8))
    fig, axis_1 = pyp.subplots(figsize= (12, 12))

    # Ploting using pyplot
    axis_1.plot(num_iters, graphs[0], label= "Output", color= 'teal', linewidth= 2)
    axis_1.set_title("Network Output Over Descent", fontsize= 16, fontweight= "bold")
    axis_1.set_xlabel("Iteration Number", fontsize= 12)
    axis_1.set_ylabel("Output Value", fontsize= 12)
    axis_1.grid(True, which= "both", linestyle= "-", linewidth= 1)
    axis_1.set_facecolor("#f0f0f0")
    axis_1.legend(loc= "best")

    pyp.tight_layout()

    # Show plot
    pyp.show()

    return fig, name

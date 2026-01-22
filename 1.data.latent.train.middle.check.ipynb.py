#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ipywidgets as widgets
from IPython.display import display, clear_output
import os
import glob
import re
from PIL import Image
import io

# --- Configuration ---

# 1. Set the path to the directory containing your images.
IMAGE_DIR = 'results_latent_simple2full'

# 2. Define the different views for the images.
VIEWS = ['xy', 'xz', 'yz']

# --- Global State ---
rotation_angle = -90
image_width = 250


# --- Dynamic Epoch Discovery ---

def find_epochs_from_files(directory):
    """
    Scans the specified directory for .png files matching the naming pattern
    and extracts all unique epoch numbers.
    """
    # Using a set to automatically handle duplicates
    epochs = set()
    # Compile a regular expression to find numbers after 'results.' and before '_'
    pattern = re.compile(r'results\.(\d+)_.*\.png')
    
    # Create the search pattern for glob
    search_path = os.path.join(directory, 'results.*.png')
    
    # Find all files matching the pattern
    for filepath in glob.glob(search_path):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            # If a match is found, add the epoch number (as an integer) to the set
            epochs.add(int(match.group(1)))
            
    # Return a sorted list of the unique epochs found
    return sorted(list(epochs))

# Automatically find epochs from the image directory
EPOCHS = find_epochs_from_files(IMAGE_DIR)


# --- Widget Setup ---

# Check if any epochs were found before creating widgets
if EPOCHS:
    # Create a selection slider for the epochs.
    epoch_slider = widgets.SelectionSlider(
        options=EPOCHS,
        value=EPOCHS[0], # Start with the first epoch
        description='Epoch',
        disabled=False,
        continuous_update=False, # Only update when the slider is released
        orientation='horizontal',
        readout=True,
        layout={'width': '50%'}
    )
    
    # Create control buttons
    refresh_button = widgets.Button(description="Refresh", icon='refresh', tooltip="Rescan folder for new epochs")
    prev_button = widgets.Button(icon='arrow-left', tooltip="Previous Epoch")
    next_button = widgets.Button(icon='arrow-right', tooltip="Next Epoch")
    rotate_button = widgets.Button(icon='rotate-right', tooltip="Rotate Images 90°")
    zoom_in_button = widgets.Button(icon='plus', tooltip="Zoom In")
    zoom_out_button = widgets.Button(icon='minus', tooltip="Zoom Out")


    # Create an output widget to hold the images. This prevents the display from flickering.
    output = widgets.Output()
else:
    # If no epochs were found, we will just display a message later.
    epoch_slider = None
    output = None

# --- Image Loading and Display Function ---

def load_and_rotate_image(filepath, not_found_label):
    """Loads an image from a file, applies the current rotation and zoom, and returns an ipywidget.Image."""
    global rotation_angle, image_width
    try:
        # Open the image using PIL
        img = Image.open(filepath)
        
        # Apply rotation if the angle is not 0. PIL rotates counter-clockwise.
        if rotation_angle != 0:
            img = img.rotate(-rotation_angle, expand=True)
        
        # Save the (potentially rotated) image to an in-memory byte buffer
        with io.BytesIO() as byte_buffer:
            img.save(byte_buffer, format='PNG')
            # Get the byte value from the buffer
            img_bytes = byte_buffer.getvalue()

        # Return the image widget using the global width
        return widgets.Image(value=img_bytes, format='png', width=image_width)
    
    except FileNotFoundError:
        return widgets.Label(not_found_label)


def view_images_for_epoch(epoch):
    """
    This function clears the output, finds the correct images for the selected epoch,
    and displays them in a 3x3 grid (Source, Target, Output).
    """
    if not output: return
    
    with output:
        # Clear the previous set of images before displaying new ones.
        clear_output(wait=True)
        
        # Lists for a 3x3 grid
        src_images_row = []
        tar_images_row = []
        output_images_row = []

        # Titles for the three rows
        src_title = widgets.HTML("<h3>Source Images</h3>")
        tar_title = widgets.HTML("<h3>Target Images</h3>")
        output_title = widgets.HTML("<h3>Output Images</h3>")

        for view in VIEWS:
            # --- Load Source Image ---
            src_file_path = os.path.join(IMAGE_DIR, f'results.{epoch}_src_{view}.png')
            src_images_row.append(load_and_rotate_image(src_file_path, f'Src not found for epoch {epoch}'))

            # --- Load Target Image ---
            tar_file_path = os.path.join(IMAGE_DIR, f'results.{epoch}_tar_{view}.png')
            tar_images_row.append(load_and_rotate_image(tar_file_path, f'Tar not found for epoch {epoch}'))
                
            # --- Load Output Image (formerly "real") ---
            output_file_path = os.path.join(IMAGE_DIR, f'results.{epoch}_{view}.png')
            output_images_row.append(load_and_rotate_image(output_file_path, f'Output not found for epoch {epoch}'))

        # Arrange the image widgets into a 3x3 grid
        grid = widgets.VBox([
            src_title,
            widgets.HBox(src_images_row),
            tar_title,
            widgets.HBox(tar_images_row),
            output_title,
            widgets.HBox(output_images_row)
        ])
        
        # Display the final grid layout.
        display(grid)

# --- Interactivity ---

def on_epoch_change(change):
    """Callback function that is triggered when the slider's value changes."""
    view_images_for_epoch(change.new)

# Click handlers for the buttons
def on_refresh_click(b):
    """Rescans the directory for epochs and updates the slider."""
    global EPOCHS
    new_epochs = find_epochs_from_files(IMAGE_DIR)
    current_value = epoch_slider.value
    
    # Update slider options without losing the current selection if it still exists
    epoch_slider.options = new_epochs
    EPOCHS = new_epochs
    if current_value in new_epochs:
        epoch_slider.value = current_value

def on_prev_click(b):
    """Moves the slider to the previous epoch."""
    current_index = EPOCHS.index(epoch_slider.value)
    if current_index > 0:
        epoch_slider.value = EPOCHS[current_index - 1]

def on_next_click(b):
    """Moves the slider to the next epoch."""
    current_index = EPOCHS.index(epoch_slider.value)
    if current_index < len(EPOCHS) - 1:
        epoch_slider.value = EPOCHS[current_index + 1]

def on_rotate_click(b):
    """Updates the rotation angle and redraws the currently viewed images."""
    global rotation_angle
    # Increment angle by 90 degrees, wrapping around at 360
    rotation_angle = (rotation_angle + 90) % 360
    # Redraw the images with the new angle
    view_images_for_epoch(epoch_slider.value)

def on_zoom_in_click(b):
    """Increases the image width and redraws."""
    global image_width
    image_width += 50
    view_images_for_epoch(epoch_slider.value)

def on_zoom_out_click(b):
    """Decreases the image width and redraws."""
    global image_width
    # Set a minimum width to prevent images from becoming invisible
    image_width = max(50, image_width - 50)
    view_images_for_epoch(epoch_slider.value)


# --- Initial Display ---

if epoch_slider and output:
    # Link the callback functions to the widgets
    epoch_slider.observe(on_epoch_change, names='value')
    refresh_button.on_click(on_refresh_click)
    prev_button.on_click(on_prev_click)
    next_button.on_click(on_next_click)
    rotate_button.on_click(on_rotate_click)
    zoom_in_button.on_click(on_zoom_in_click)
    zoom_out_button.on_click(on_zoom_out_click)
    
    # Create a layout for the controls
    controls = widgets.HBox([prev_button, epoch_slider, next_button, refresh_button, rotate_button, zoom_out_button, zoom_in_button])
    
    # Display the main container with the controls and the output area.
    display(widgets.VBox([controls, output]))

    # Trigger the display for the initial value of the slider.
    view_images_for_epoch(epoch_slider.value)
else:
    # If no valid image files were found, display an error message.
    error_message = f"No images found in '{IMAGE_DIR}'. Please check the path and that filenames match the 'results.EPOCH_NUMBER_....png' format."
    display(widgets.HTML(f'<p style="color:red;">{error_message}</p>'))


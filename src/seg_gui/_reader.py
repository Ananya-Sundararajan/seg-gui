'''
from pathlib import Path

import dask.array as da
import imageio.v3 as iio
import numpy as np
from dask import delayed
from natsort import natsorted

# Define the expected folder names
IMAGE_FOLDER_NAMES = ["dapi", "bg", "dapi_bg", "images"]
MASK_FOLDER_NAME = "masks"
# Supported extensions
EXTENSIONS = {".tif", ".tiff", ".png", ".TIF", ".TIFF", ".PNG"}


def napari_get_reader(path):
    path = Path(path)
    if not path.is_dir():
        return None
    return reader_function


def safe_read(file_path):
    """
    Reads an image and ensures it is returned as a 2D array (H, W).
    If the image is RGB/RGBA, it extracts the first channel.
    """
    img = iio.imread(file_path)
    # If the image has channels (H, W, C), take only the first one
    if img.ndim == 3:
        return img[..., 0]
    return img


def reader_function(path) -> list[tuple[da.Array, dict, str]]:
    path = Path(path)
    all_inputs: list[tuple[Path, str, str]] = []

    # 1. Identify all existing folders
    for name in IMAGE_FOLDER_NAMES:
        image_dir = path / name
        if image_dir.exists():
            all_inputs.append((image_dir, name, "image"))

    mask_dir = path / MASK_FOLDER_NAME
    if mask_dir.exists():
        all_inputs.append((mask_dir, MASK_FOLDER_NAME, "labels"))

    # --- Validation ---
    if not any(t[2] == "image" for t in all_inputs) or not any(
        t[2] == "labels" for t in all_inputs
    ):
        raise ValueError(
            f"Input directory must contain an image folder ({IMAGE_FOLDER_NAMES}) "
            f"and a '{MASK_FOLDER_NAME}' folder."
        )

    def get_files(directory):
        files = []
        for ext in EXTENSIONS:
            files.extend(list(directory.glob(f"*{ext}")))
        if not files:
            raise ValueError(f"No valid images found in {directory.name}/")
        return natsorted(files)

    # 2. Determine metadata from the first image
    first_image_input = [t for t in all_inputs if t[2] == "image"][0]
    image_files = get_files(first_image_input[0])

    # We use safe_read here to ensure our metadata reflects a 2D slice
    sample_img = safe_read(image_files[0])
    metadata_shape = sample_img.shape  # This will now be (H, W)
    metadata_dtype = sample_img.dtype

    # 3. Create Dask Stacks
    dask_stacks_to_return = []

    for directory, name, layer_type in all_inputs:
        current_files = get_files(directory)

        if len(current_files) != len(image_files):
            raise ValueError(
                f"File count mismatch in '{name}/'. Expected {len(image_files)}."
            )

        # Force integer types for label layers
        current_dtype = np.uint16 if layer_type == "labels" else metadata_dtype

        dask_chunks = []
        for f in current_files:
            # Wrap the reading logic in the delayed safe_read function
            delayed_read = delayed(safe_read)(f)
            dask_chunks.append(
                da.from_delayed(
                    delayed_read, shape=metadata_shape, dtype=current_dtype
                )
            )

        # Stack slices into a (N, H, W) dask array
        stack = da.stack(dask_chunks, axis=0)
        dask_stacks_to_return.append(
            (stack, {"name": f"{name}_stack"}, layer_type)
        )

    return dask_stacks_to_return
'''
import numpy as np
import imageio.v3 as iio
from pathlib import Path
import dask.array as da
from dask import delayed
from natsort import natsorted

# Supported extensions
EXTENSIONS = {".tif", ".tiff", ".png", ".TIF", ".TIFF", ".PNG"}

def napari_get_reader(path):
    """Entry point for napari to recognize this plugin as a reader."""
    if isinstance(path, list):
        path = path[0]
    path = Path(path)
    
    # If the path is a directory, we return our reader function
    if path.is_dir():
        return reader_function
    return None

def safe_read(file_path):
    """
    Standardizes image loading across platforms.
    Ensures output is 2D (H, W). Handles RGB TIFFs/PNGs by taking one channel.
    """
    img = iio.imread(file_path)
    
    # 1. Remove singleton dimensions (e.g., (1, H, W) -> (H, W))
    img = np.squeeze(img)
    
    # 2. Handle Multichannel/RGB (H, W, 3) or (H, W, 4)
    if img.ndim == 3:
        # If the last dimension is 3 or 4, it's likely RGB/A
        if img.shape[-1] in [3, 4]:
            return img[..., 0]
        # If the first dimension is 3 or 4, it's likely Channel-first
        elif img.shape[0] in [3, 4]:
            return img[0, ...]
            
    return img

def get_files(directory):
    """
    Finds valid image files while ignoring hidden system files like .DS_Store
    or Windows thumbnail files.
    """
    files = []
    for ext in EXTENSIONS:
        # Ignore hidden files starting with '.'
        valid_files = [
            f for f in directory.glob(f"*{ext}") 
            if not f.name.startswith(".")
        ]
        files.extend(valid_files)
    
    if not files:
        raise ValueError(f"No valid images found in {directory}")
    
    # Natural sorting ensures 'image_2.tif' comes before 'image_10.tif'
    return natsorted(files)

def reader_function(path):
    """The actual function that loads the data into napari layers."""
    path = Path(path)
    results = []

    # 1. Define folder mapping: (folder_name, layer_name, layer_type)
    folder_map = [
        ("dapi", "DAPI", "image"),
        ("bg", "Background", "image"),
        ("images", "Images", "image"),
        ("masks", "Masks", "labels")
    ]

    # 3. Build Dask stacks for each existing folder
    for folder_name, layer_name, layer_type in folder_map:
        dir_path = path / folder_name

        if not dir_path.exists():
            continue

        current_files = get_files(dir_path)
        if not current_files:
            continue

        # 🔑 FIX: detect shape + dtype PER FOLDER
        first_img = safe_read(current_files[0])
        metadata_shape = first_img.shape
        metadata_dtype = first_img.dtype

        dask_chunks = []
        for f in current_files:
            delayed_read = delayed(safe_read)(f)
            dask_chunks.append(
                da.from_delayed(
                    delayed_read,
                    shape=metadata_shape,
                    dtype=metadata_dtype,
                )
            )

        # Stack all images into a single 3D array (Z, H, W)
        stack = da.stack(dask_chunks, axis=0)

        results.append((stack, {"name": layer_name}, layer_type))

    return results
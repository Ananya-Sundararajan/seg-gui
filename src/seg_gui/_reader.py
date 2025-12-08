from pathlib import Path

import dask.array as da
import imageio.v3 as iio
import numpy as np
from dask import delayed
from natsort import natsorted

# Define the expected folder names for the image channels
IMAGE_FOLDER_NAMES = [
    "dapi",
    "bg",
    "dapi_bg",
    "images",
]  # Added "images" for potential backward compatibility
MASK_FOLDER_NAME = "masks"


def napari_get_reader(path):

    path = Path(path)

    if not path.is_dir():
        return None

    return reader_function


def reader_function(path) -> list[tuple[da.Array, dict, str]]:

    path = Path(path)

    # 1. Identify all existing folders and collect metadata

    # Store tuples of (folder_path, layer_name, layer_type)
    all_inputs: list[tuple[Path, str, str]] = []

    # Check for image folders, prioritizing DAPI/BG over generic 'images'
    for name in IMAGE_FOLDER_NAMES:
        image_dir = path / name
        if image_dir.exists():
            # If multiple image folders exist, all are added here.
            all_inputs.append((image_dir, name, "image"))

    # Check for the mandatory mask folder
    mask_dir = path / MASK_FOLDER_NAME
    if mask_dir.exists():
        all_inputs.append((mask_dir, MASK_FOLDER_NAME, "labels"))

    # --- Validation ---
    num_image_stacks = len([t for t in all_inputs if t[2] == "image"])
    has_mask_stack = any(t[2] == "labels" for t in all_inputs)

    if num_image_stacks == 0 or not has_mask_stack:
        raise ValueError(
            f"Input directory must contain at least one image folder "
            f"({', '.join(IMAGE_FOLDER_NAMES)}) and the '{MASK_FOLDER_NAME}' folder."
        )

    # 2. Determine metadata (shape/dtype) from the first image stack found

    first_image_input = [t for t in all_inputs if t[2] == "image"][0]
    first_image_dir = first_image_input[0]

    def get_files(directory):
        # Find the first file's extension to determine the file type
        # try:
        ext = next(directory.glob("*.*")).suffix
        # except StopIteration:
        #    raise ValueError(f"Subfolder '{directory.name}/' is empty.")

        files = natsorted(list(directory.glob(f"*{ext}")))
        if not files:
            raise ValueError(
                f"No files found in {directory.name}/ with extension {ext}."
            )
        return files, ext

    # Load the first image to define the required shape/dtype metadata for Dask
    image_files, _ = get_files(first_image_dir)
    first_image = iio.imread(image_files[0])
    metadata_shape = first_image.shape
    metadata_dtype = first_image.dtype

    # 3. Create Dask Stacks for ALL found folders
    dask_stacks_to_return = []

    for directory, name, layer_type in all_inputs:

        current_files, current_ext = get_files(directory)

        # All images must have the same number of slices/files
        if len(current_files) != len(image_files):
            raise ValueError(
                f"File count mismatch: '{name}/' has {len(current_files)} files, "
                f"but expected {len(image_files)} (from {first_image_dir.name}/)."
            )

        # Use the metadata from the first image stack for all stacks
        current_dtype = metadata_dtype
        current_shape = metadata_shape

        # For mask data, we must ensure the dtype is suitable for labels (integers)
        if layer_type == "labels":
            # Use uint16 as a safe default for labels unless the image is clearly smaller
            current_dtype = np.uint16
            current_shape = (
                metadata_shape  # Shape remains the same (H, W) or (H, W, 1)
            )

        dask_chunks = []
        for f in current_files:
            # Create a Dask array chunk for each file, lazy loading
            delayed_read = delayed(iio.imread)(f)
            dask_chunks.append(
                da.from_delayed(
                    delayed_read, shape=current_shape, dtype=current_dtype
                )
            )

        # Stack the list of Dask chunks into one large Dask array (still LAZYLY)
        stack = da.stack(dask_chunks, axis=0)

        # 4. Define napari layer data structure
        layer_data = (stack, {"name": f"{name}_stack"}, layer_type)
        dask_stacks_to_return.append(layer_data)

    # This list will contain the DAPI stack(s) first, followed by the MASK stack.
    return dask_stacks_to_return

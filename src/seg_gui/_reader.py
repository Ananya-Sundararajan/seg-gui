from pathlib import Path
from typing import List, Tuple

import dask.array as da
import imageio.v3 as iio
import numpy as np
from dask import delayed
from natsort import natsorted

IMAGE_FOLDER_NAMES = [
    "dapi",
    "bg",
    "dapi_bg",
    "images",  # backward compatibility
]

MASK_FOLDER_NAME = "masks"

VALID_EXTENSIONS = {".tif", ".tiff", ".png"}

def napari_get_reader(path):
    path = Path(path)
    if path.is_dir():
        return reader_function
    return None

def safe_read(path: Path) -> np.ndarray:
    """
    Read an image and normalize it to a 2D NumPy array (H, W).

    Handles:
    - singleton dimensions
    - RGB / RGBA images
    - platform-specific imageio behavior
    """

    img = iio.imread(path)
    img = np.asarray(img)
    img = np.squeeze(img)

    # Handle RGB / RGBA
    if img.ndim == 3:
        # channel-last
        if img.shape[-1] in (3, 4):
            img = img[..., 0]
        # channel-first
        elif img.shape[0] in (3, 4):
            img = img[0, ...]

    if img.ndim != 2:
        raise ValueError(
            f"{path.name} has unsupported shape {img.shape}. "
            "Only 2D images are supported."
        )

    return img

def get_files(directory: Path) -> List[Path]:
    files = [
        f
        for f in directory.iterdir()
        if f.is_file()
        and f.suffix.lower() in VALID_EXTENSIONS
        and not f.name.startswith(".")
    ]

    files = natsorted(files)

    if not files:
        raise ValueError(f"No valid image files found in '{directory.name}/'")

    return files


def reader_function(path) -> List[Tuple[da.Array, dict, str]]:
    path = Path(path)

    # Discover folders
    inputs: List[Tuple[Path, str, str]] = []

    for name in IMAGE_FOLDER_NAMES:
        image_dir = path / name
        if image_dir.exists():
            inputs.append((image_dir, name, "image"))

    mask_dir = path / MASK_FOLDER_NAME
    if mask_dir.exists():
        inputs.append((mask_dir, MASK_FOLDER_NAME, "labels"))

    num_image_stacks = sum(1 for _, _, t in inputs if t == "image")
    has_masks = any(t == "labels" for _, _, t in inputs)

    if num_image_stacks == 0 or not has_masks:
        raise ValueError(
            f"Input directory must contain at least one image folder "
            f"({', '.join(IMAGE_FOLDER_NAMES)}) and a '{MASK_FOLDER_NAME}' folder."
        )

    # Establish reference metadata from FIRST IMAGE STACK
    first_image_dir = next(p for p, _, t in inputs if t == "image")
    image_files = get_files(first_image_dir)

    first_img = safe_read(image_files[0])
    metadata_shape = first_img.shape
    metadata_dtype = first_img.dtype
    expected_length = len(image_files)

    # Build Dask stacks
    results = []

    for directory, name, layer_type in inputs:
        current_files = get_files(directory)

        if len(current_files) != expected_length:
            raise ValueError(
                f"File count mismatch in '{directory.name}/': "
                f"{len(current_files)} files found, expected {expected_length}."
            )

        # Labels must be integer dtype
        if layer_type == "labels":
            current_dtype = np.uint16
        else:
            current_dtype = metadata_dtype

        dask_chunks = []
        for f in current_files:
            delayed_read = delayed(safe_read)(f)
            dask_chunks.append(
                da.from_delayed(
                    delayed_read,
                    shape=metadata_shape,
                    dtype=current_dtype,
                )
            )

        stack = da.stack(dask_chunks, axis=0)

        metadata = {
            "name": f"{name}_stack",
            "filenames": [f.name for f in current_files] 
        }

        results.append((stack, metadata, layer_type))

    return results

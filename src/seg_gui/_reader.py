"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

from pathlib import Path

import imageio
import numpy as np
import tifffile as tiff
from natsort import natsorted


def napari_get_reader(path):
    """An implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    path = Path(path)

    # if we know we cannot read the file, we immediately return None.
    if not path.is_dir():
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    path = Path(path)

    # paths
    image_dir = path / "images"
    mask_dir = path / "masks"

    if not image_dir.exists() or not mask_dir.exists():
        raise ValueError(
            "Input directory must contain 'images/' and 'masks/' subfolders."
        )

    image_ext = next(
        image_dir.glob("*.*")
    ).suffix  # take the first file's suffix
    mask_ext = next(mask_dir.glob("*.*")).suffix

    # Collect files
    image_files = (
        natsorted(list(image_dir.glob("*.tif")))
        if image_ext == ".tif"
        else natsorted(list(image_dir.glob("*.png")))
    )
    mask_files = (
        natsorted(list(mask_dir.glob("*.tif")))
        if mask_ext == ".tif"
        else natsorted(list(mask_dir.glob("*.png")))
    )

    if len(image_files) == 0:
        raise ValueError("No .tif or .png files found in images/.")
    if len(mask_files) == 0:
        raise ValueError("No .tif or .png files found in masks/.")

    # Load into stacks
    if image_ext == ".tif":
        images = [tiff.imread(f) for f in image_files]
    else:
        images = [imageio.imread(f) for f in image_files]

    if mask_ext == ".tif":
        masks = [tiff.imread(f) for f in mask_files]
    else:
        masks = [imageio.imread(f) for f in mask_files]

    images_stack = np.stack(images)
    masks_stack = np.stack(masks)

    return [
        (images_stack, {"name": "images_stack"}, "image"),
        (masks_stack, {"name": "masks_stack"}, "labels"),
    ]

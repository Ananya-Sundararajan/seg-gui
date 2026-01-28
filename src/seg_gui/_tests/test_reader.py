import numpy as np
import imageio.v3 as iio
from seg_gui import napari_get_reader
from pathlib import Path

def test_reader(tmp_path):
    # 1. Setup: Create the folder structure your reader expects
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # 2. Create fake image and mask files (must have same name and count)
    test_name = "test_01.tif"
    fake_image = np.zeros((10, 10), dtype=np.uint8)
    fake_mask = np.zeros((10, 10), dtype=np.uint16)

    iio.imwrite(img_dir / test_name, fake_image)
    iio.imwrite(mask_dir / test_name, fake_mask)

    # 3. Test: Get the reader
    # napari_get_reader should return a function when given a directory
    reader = napari_get_reader(str(tmp_path))
    assert callable(reader)

    # 4. Test: Run the reader
    layer_data_list = reader(str(tmp_path))

    # We expect 2 layers: one for 'images' and one for 'masks'
    assert isinstance(layer_data_list, list)
    assert len(layer_data_list) == 2

    # Verify the image layer data
    image_data = layer_data_list[0][0]
    assert image_data.shape == (1, 10, 10)  # (Stack size, H, W)

    # Verify filenames were captured in metadata (as we added to your reader)
    metadata = layer_data_list[1][1] # Mask layer metadata
    assert "filenames" in metadata
    assert metadata["filenames"][0] == test_name

def test_get_reader_returns_none_for_file(tmp_path):
    # Your reader specifically returns None if the path is a file, not a dir
    fake_file = tmp_path / "not_a_folder.tif"
    fake_file.write_text("dummy")
    
    reader = napari_get_reader(str(fake_file))
    assert reader is None
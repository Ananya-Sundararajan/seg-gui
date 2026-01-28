import numpy as np
import pytest
from seg_gui._widget import SegmentationEditor

def test_segmentation_editor_init(make_napari_viewer):
    """
    Test that our SegmentationEditor widget can be initialized 
    without crashing.
    """
    # 1. Create a viewer using the napari pytest fixture
    viewer = make_napari_viewer()
    
    # 2. Add a dummy image layer so the widget has something to look at
    viewer.add_image(np.random.random((100, 100)), name="test_image")

    # 3. Initialize the widget
    widget = SegmentationEditor(viewer)

    # 4. Simple assertion to check it exists
    assert widget is not None
    assert widget.viewer == viewer
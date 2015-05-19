import os
from dsi2.config import dsi2_data_path

"""
Tests that the data included with DSI2 is available
"""

def test_initialization():
    assert os.path.exists(dsi2_data_path)

def test_example_data():
    # Check for the lausanne graphml files
    lausanne_paths = [
        "lausanne2008/resolution83/resolution83.graphml",
        "lausanne2008/resolution150/resolution150.graphml",
        "lausanne2008/resolution258/resolution258.graphml",
        "lausanne2008/resolution500/resolution500.graphml",
        "lausanne2008/resolution1015/resolution1015.graphml"]
    for pth in lausanne_paths:
        assert os.path.exists(
            os.path.join(dsi2_data_path,pth))
    
    # check for the mni152
    from dsi2.volumes import get_MNI152_path
    assert os.path.exists(get_MNI152_path())

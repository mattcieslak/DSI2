import os

os.environ['DSI2_DATA'] = 'DSI2'
os.environ['LOCAL_TRACKDB'] = 'TRACKDB'

import dsi2.config

def test_initialization():
    assert dsi2.config.dsi2_data_path == 'DSI2'
    assert dsi2.config.local_trackdb_path == 'TRACKDB'

def test_loadConfig():
    os.environ['DSI2_DATA'] = 'NEWDSI2'
    os.environ['LOCAL_TRACKDB'] = 'NEWTRACKDB'
    dsi2.config.loadConfig()
    assert dsi2.config.dsi2_data_path == 'DSI2'
    assert dsi2.config.local_trackdb_path == 'TRACKDB'
    dsi2.config.loadConfig(True)
    assert dsi2.config.dsi2_data_path == 'NEWDSI2'
    assert dsi2.config.local_trackdb_path == 'NEWTRACKDB'

def test_undefinedEnv():
    del os.environ['DSI2_DATA']
    del os.environ['LOCAL_TRACKDB']
    dsi2.config.loadConfig(True)
    assert dsi2.config.dsi2_data_path == 'example_data'
    assert dsi2.config.local_trackdb_path == 'example_trackdb'


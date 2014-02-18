import os, logging

default_dsi2_data_path = "example_data"
default_local_trackdb_path = "example_trackdb"
logfile = "dsi2.log"

FORMAT = '%(asctime)s %(levelname)s %(pathname)s:%(lineno)s %(message)s'
logging.basicConfig(format = FORMAT, filename = logfile)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config_loaded = False
dsi2_data_path = ""
local_trackdb_path = ""

def loadConfig(force = False):
    global config_loaded, dsi2_data_path, local_trackdb_path
    if (not config_loaded or force):
      
        config_loaded = True
        envVar = 'DSI2_DATA'
        dsi2_data_path = os.getenv(envVar)

        if (dsi2_data_path == None):
            dsi2_data_path = default_dsi2_data_path
            logger.warning('%s not defined. Using %s', envVar, dsi2_data_path)
        else:
            logger.info('using %s from %s', dsi2_data_path, envVar)

        envVar = 'LOCAL_TRACKDB'
        local_trackdb_path = os.getenv(envVar)
      
        if (local_trackdb_path == None):
            local_trackdb_path = default_local_trackdb_path
            logger.warning('%s is not defined', envVar, local_trackdb_path)
        else:
            logger.info('using %s from %s', local_trackdb_path, envVar)

loadConfig()

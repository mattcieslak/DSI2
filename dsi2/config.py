import os, logging
logfile = "dsi2.log"

FORMAT = '%(asctime)s %(levelname)s %(pathname)s:%(lineno)s %(message)s'
logging.basicConfig(format = FORMAT, filename = logfile)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config_loaded = False
dsi2_data_path = ""

from pkg_resources import Requirement, resource_filename

def loadConfig(force = False):
    global config_loaded, dsi2_data_path
    if (not config_loaded or force):

        using_pkg_resources = True
        try:
            dsi2_data_path = resource_filename(
                   Requirement.parse("dsi2"),"dsi2/example_data")
        except Exception:
            using_pkg_resources = False

        if not using_pkg_resources:
            if not os.path.exists("/Applications/DSI2.app"):
                raise OSError("Unable to locate DSI2 data resources")
            dsi2_data_path = "/Applications/DSI2.app/Contents/Resources/dsi2/example_data"
        config_loaded=True



loadConfig()

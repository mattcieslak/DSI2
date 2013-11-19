import sys
sys.path.append("..")
from dsi2.ui.local_data_importer import LocalDataImporter

example_folder = "/home/cieslak/example_dsi2_input_data"
example_trackdb_json = "/home/cieslak/example_trackdb/example_data.json"

ldi=LocalDataImporter(json_file=example_trackdb_json)


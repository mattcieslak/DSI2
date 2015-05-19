import os
""" 
NOTE: the testing data should be located in a directory in your home
directory.

Required data:
---------------
~/testing_data/testing_input
~/testing_data/testing_output

"""
# Directories where input and output data will go
home = os.path.expanduser("~")
test_input_data = os.path.join(home,"dsi2_testing_data", "testing_input")
test_output_data = os.path.join(home,"dsi2_testing_data", "testing_output")

# Json data describing the files to be "imported" for testing
input_data_json = os.path.join(test_input_data,"test_data.json")
local_trackdb_dir = os.path.join(test_output_data,"local_trackdb")
local_trackdb_dir = os.path.join(test_output_data,"local_mongodb")
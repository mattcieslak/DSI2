import os
""" 
NOTE: the testing data should be located in a directory in your home
directory.

Required data:
---------------
~/testing_data/testing_input
~/testing_data/testing_output

"""

home = os.getenv("HOME")
test_input_data = os.path.join(home,"testing_data", "testing_input")
test_output_data = os.path.join(home,"testing_data", "testing_output")

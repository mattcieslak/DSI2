#!/usr/bin/env python
from setuptools import setup, find_packages
from codecs import open
from os import path

# Version
MAJOR="0"
MINOR="5"
MICRO="0"
VERSION= MAJOR + "." + MINOR + "." + MICRO

here = path.abspath(path.dirname(__file__))
with open(path.join(here,'README.md')) as f:
    long_description = f.read()

setup(  name='DSI2',
        version=VERSION,
        description='DSI2 Toolbox',
        author='Matthew Cieslak',
        author_email='mattcieslak@gmail.com',
        license='GPLv3',
        url='https://github.com/mattcieslak/DSI2',
        packages = find_packages( exclude = ['doc', 'tests', "example_data"]),
        include_package_data = True, # Comes from MANIFEST.in
        entry_points = {
            'gui_scripts':[
                'dsi2_browse = dsi2.app_launch:browser_builder',
                'dsi2_import = dsi2.app_launch:import_data',
                'dsi2_view = dsi2.app_launch:view_tracks'
            ]
        }
)


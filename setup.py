#!/usr/bin/env python
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here,'README.md')) as f:
    long_description = f.read()

setup(  name='DSI2',
        version='0.2.0',
        description='DSI2 Toolbox',
        author='Matthew Cieslak',
        author_email='mattcieslak@gmail.com',
        license='GPLv3',
        url='https://github.com/mattcieslak/DSI2',
        packages = find_packages( exclude = ['doc', 'tests']),
        include_package_data = True,
        package_data={
            "dsi2":[
                "dsi2/example_data/lausanne2008/ParcellationLausanne2008.xls",
                "dsi2/example_data/lausanne2008/README.txt",
                "dsi2/example_data/lausanne2008/resolution1015/resolution1015.graphml",
                "dsi2/example_data/lausanne2008/resolution150/resolution150.graphml",
                "dsi2/example_data/lausanne2008/resolution258/resolution258.graphml",
                "dsi2/example_data/lausanne2008/resolution500/resolution500.graphml",
                "dsi2/example_data/lausanne2008/resolution83/resolution83.graphml",
                "dsi2/example_data/MNI152_T1_2mm_brain_mask.nii.gz",
                "dsi2/example_data/MNI152_T1_2mm.nii.gz",
                "dsi2/example_data/MNI_BRAIN_MASK_FLOAT.nii.gz",
                "dsi2/example_data/NTU90_QA.nii.gz"
            ]
        },
        entry_points = {
            'gui_scripts':[
                'dsi2_browse = dsi2.app_launch:browser_builder',
                'dsi2_import = dsi2.app_launch:import_data',
                'dsi2_view = dsi2.app_launch:view_tracks'
            ]
        }
)


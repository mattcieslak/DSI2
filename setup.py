#!/usr/bin/env python
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here,'README.md')) as f:
    long_description = f.read()

setup(  name='DSI2',
        version='0.1.0',
        description='DSI2 Toolbox',
        author='Matthew Cieslak',
        author_email='mattcieslak@gmail.com',
        license='GPLv3',
        url='https://github.com/mattcieslak/DSI2',
        packages = find_packages( exclude = ['doc', 'tests']),
        package_data={
            "":[
                "example_data/lausanne2008/ParcellationLausanne2008.xls",
                "example_data/lausanne2008/README.txt",
                "example_data/lausanne2008/resolution1015/resolution1015.graphml",
                "example_data/lausanne2008/resolution150/resolution150.graphml",
                "example_data/lausanne2008/resolution258/resolution258.graphml",
                "example_data/lausanne2008/resolution500/resolution500.graphml",
                "example_data/lausanne2008/resolution83/resolution83.graphml",
                "example_data/MNI152_T1_2mm_brain_mask.nii.gz",
                "example_data/MNI152_T1_2mm.nii.gz",
                "example_data/MNI_BRAIN_MASK_FLOAT.nii.gz",
                "example_data/NTU90_QA.nii.gz"
            ]
        },
        scripts = [
            "scripts/launch_browser_builder.py"
            ]
            #'gui_scripts':[
            #    'dsi2_browse=scripts:broswer_builder'
            #]
        #}
)


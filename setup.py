#!/usr/bin/env python
from distutils.core import setup

setup(name='DSI2',
        version='0.1',
        description='DSI2 Toolbox',
        author='Matthew Cieslak',
        author_email='mattcieslak@gmail.com',
        url='https://github.com/mattcieslak/DSI2',
        packages = ['dsi2', 'dsi2/aggregation', 'dsi2/database', 'dsi2/streamlines', 'dsi2/ui', 'dsi2/volumes'],
       )


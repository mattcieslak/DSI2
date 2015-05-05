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


# Variables for py2app
p2a_includes=[
    'PyQt4',
    'subprocess', 
    'numpy',
    'numpy.core',
    'configobj',
    'scipy',

    'traits',
    'traitsui',
    'traitsui.editors',
    'traitsui.editors.*',
    'traitsui.extras',
    'traitsui.extras.*',
    'traitsui.image',
    'traitsui.image.*',
    'traitsui.ui_traits',

    'traits.api',
    'traits.*',

    'vtk',
    'tvtk',
    'tvtk.*',
    'tvtk.tvtk_classes',
    'tvtk.pyface.*',
    'tvtk.pyface.ui.qt4',
    'tvtk.pyface.ui.qt4.*',
    'tvtk.tools',
    'tvtk.tools.*',
    'tvtk.view',
    'tvtk.view.*',
    'tvtk.plugins',
    'tvtk.plugins.*',

    'traitsui.qt4',
    'traitsui.qt4.*',
    'chaco',
    'chaco.*',

    'kiva',

    'pyface',
    'pyface.*',
    'pyface.qt4',
    'pyface.toolkit',
    'pyface.image_resource',
    'pyface.image_resource.*',

    'pyface.ui.qt4',
    'pyface.ui.qt4.init',
    'pyface.ui.qt4.*',
    'pyface.ui.qt4.grid.*',
    'pyface.ui.qt4.action.*',
    'pyface.ui.qt4.timer.*',
    'pyface.ui.qt4.wizard.*',
    'pyface.ui.qt4.workbench.*',

    'enable',
    'enable.drawing',
    'enable.tools',
    'enable.qt4',
    'enable.qt4.*',

    'enable.savage',
    'enable.savage.*',
    'enable.savage.svg',
    'enable.savage.svg.*',
    'enable.savage.svg.backends',
    #'enable.savage.svg.backends.wx',
    #'enable.savage.svg.backends.wx.*',
    'enable.savage.svg.css',
    'enable.savage.compliance',
    'enable.savage.trait_defs',
    'enable.savage.trait_defs.*',
    'enable.savage.trait_defs.ui',
    'enable.savage.trait_defs.ui.*',
    'enable.savage.trait_defs.ui.qt4',
    'enable.savage.trait_defs.ui.qt4.*',

    'matplotlib',
    #'dsi2'
    #'dsi2.*'
    #'dsi2.aggregation.segmentation'

    'sklearn',
    'sklearn.metrics',
    'sklearn.cluster',
    'sklearn.utils.lgamma',
    'sklearn.utils.sparsetools.*',
    'sklearn.neighbors.*'
]


p2a_packages=[]
p2a_options = dict(
      includes=p2a_includes,
      packages=p2a_packages,
      excludes=["/usr/local/Cellar/vtk5/5.10.1_1/lib/vtk-5.10/libvtkIOPythonD.5.10.dylib"],#anything we need to forcibly exclude?
      #resources=resources,
      argv_emulation=True,
      site_packages=True,
      #frameworks=frameworks,
      iconfile='dsi2/example_data/dsi2.icns',
      plist=dict(
	  CFBundleIconFile='dsi2/example_data/dsi2.icns',
	  CFBundleName               = "DSI2",
	  CFBundleShortVersionString = VERSION,     # must be in X.X.X format
	  CFBundleGetInfoString      = "DSI2 "+ VERSION,
	  CFBundleExecutable         = "DSI2view",
	  CFBundleIdentifier         = "org.dsi2.DSI2view",
	  CFBundleLicense            = "GNU GPLv3+",
	  CFBundleDocumentTypes=[{"CFBundleTypeExtensions":['*']}],
	  )
  )


setup(  
        app=['DSI2app.py'],
        name='DSI2',
        version=VERSION,
        description='DSI2 Toolbox',
        author='Matthew Cieslak',
        author_email='mattcieslak@gmail.com',
        license='GPLv3',
        url='https://github.com/mattcieslak/DSI2',
        packages = find_packages( exclude = ['doc', 'tests', "example_data"]),
        include_package_data = True, # Comes from MANIFEST.in
        package_data={
            "dsi2":[
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
            ]},
        options = {"py2app":p2a_options},
        setup_requires = ['py2app'],
        entry_points = {
            'gui_scripts':[
                'dsi2_browse = dsi2.app_launch:browser_builder',
                'dsi2_import = dsi2.app_launch:import_data',
                'dsi2_view = dsi2.app_launch:view_tracks'
            ]
	}
)


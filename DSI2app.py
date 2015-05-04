#!/usr/bin/env python
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

from dsi2.ui.sphere_browser import SphereBrowser
sb = SphereBrowser()
sb.configure_traits()

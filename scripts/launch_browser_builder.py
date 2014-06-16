#!/usr/bin/env python
import time, sys, os

from dsi2.ui.browser_builder import BrowserBuilder

from dsi2.database.local_data import get_local_data
local_json = os.path.join(os.getenv("HOME"),
                          "example_trackdb","example_data.json")
bb = BrowserBuilder(local_json = local_json)
bb.configure_traits()



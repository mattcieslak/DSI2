#!/usr/bin/env python

from traits.api import HasTraits, Instance, Array, Bool, Dict, \
     on_trait_change, Delegate, List, Color, Any, Instance, Int, File, \
     Button, Enum, Str, DelegatesTo, Property
from traitsui.api import View, Item, HGroup, VGroup, \
     Group, Handler, HSplit, VSplit, RangeEditor, Include, Action, MenuBar, Menu, \
     TableEditor, ObjectColumn, Separator

from traitsui.extras.checkbox_column import CheckboxColumn

from ..volumes.roi import ROI
from ..volumes.scalar_volume import ScalarVolumes
from ..database.traited_query import Scan
from dsi2.volumes import get_MNI152_path

from tvtk.pyface.scene import Scene
from mayavi.core.ui.api import SceneEditor


from traitsui.editors.tabular_editor import TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from traitsui.file_dialog import save_file, open_file
from tvtk.pyface.scene import Scene
from tvtk.api import tvtk

from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel
import os
import numpy as np


class LTPAResults(HasTraits):
    results = List(Instance(LTPAResult))

class LTPAResult(HasTraits):
    # 3d MayaVi scene that will display slices and streamlines
    scene3d = Instance(MlabSceneModel)

    # 
    result_coords = Array
    track_setsA = List(Instance(TrackDataset))
    track_setsB = List(Instance(TrackDataset))


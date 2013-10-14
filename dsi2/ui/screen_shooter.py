#!/usr/bin/env python

from traits.api import HasTraits, Instance, Bool, Dict, \
     on_trait_change, Color, Instance, File, Int
from traitsui.api import View, Item, OKButton, CancelButton

from tvtk.pyface.scene import Scene

from mayavi.core.ui.api import SceneEditor
from mayavi.tools.mlab_scene_model import MlabSceneModel

class ScreenShooter(HasTraits):
    DPI = Int(170)
    magnification = Int(4)
    scene3d = Instance(MlabSceneModel)
    filepath = File

    traits_view = View(
                     Item("DPI"),
                     Item("magnification"),
                     Item("filepath"),
                     kind="modal",buttons=[OKButton,CancelButton]
                 )
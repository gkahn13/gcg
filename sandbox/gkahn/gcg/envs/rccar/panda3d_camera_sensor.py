import sys
import numpy as np
from panda3d.core import WindowProperties, FrameBufferProperties
from panda3d.core import GraphicsPipe, GraphicsEngine, GraphicsOutput
from panda3d.core import Texture

class Panda3dCameraSensor(object):
    def __init__(self, base, color=True, depth=False, size=None, near_far=None, hfov=None, title=None):
        if size is None:
            size = (640, 480)
        if near_far is None:
            near_far = (1.0, 1000.0)
        if hfov is None:
            hfov = 60
        winprops = WindowProperties.size(*size)
        winprops.setTitle(title or 'Camera Sensor')
        fbprops = FrameBufferProperties()
        # Request 8 RGB bits, 8 alpha bits, and a depth buffer.
        fbprops.setRgbColor(True)
        fbprops.setRgbaBits(8, 8, 8, 8)
        fbprops.setDepthBits(24)
        self.graphics_engine = GraphicsEngine(base.pipe)

        window_type = base.config.GetString('window-type', 'onscreen')
        flags = GraphicsPipe.BFFbPropsOptional
        if window_type == 'onscreen':
            flags = flags | GraphicsPipe.BFRequireWindow
        elif window_type == 'offscreen':
            flags = flags | GraphicsPipe.BFRefuseWindow

        self.buffer = self.graphics_engine.makeOutput(
            base.pipe, "camera sensor buffer", 0,
            fbprops, winprops, flags)

        if not color and not depth:
            raise ValueError("At least one of color or depth should be True")
        if color:
            self.color_tex = Texture("color_texture")
            self.buffer.addRenderTexture(self.color_tex, GraphicsOutput.RTMCopyRam,
                                         GraphicsOutput.RTPColor)
        else:
            self.color_tex = None
        if depth:
            self.depth_tex = Texture("depth_texture")
            self.buffer.addRenderTexture(self.depth_tex, GraphicsOutput.RTMCopyRam,
                                         GraphicsOutput.RTPDepth)
        else:
            self.depth_tex = None

        self.cam = base.makeCamera(self.buffer, scene=base.render, camName='camera_sensor')
        self.lens = self.cam.node().getLens()
        self.lens.setFov(hfov)
        self.lens.setFilmSize(*size)  # this also defines the units of the focal length
        self.lens.setNearFar(*near_far)

    def observe(self):
        for _ in range(self.graphics_engine.getNumWindows()):
            self.graphics_engine.renderFrame()
        self.graphics_engine.syncFrame()

        images = []

        if self.color_tex:
            data = self.color_tex.getRamImageAs('RGBA')
            if sys.version_info < (3, 0):
                data = data.get_data()
            image = np.frombuffer(data, np.uint8)
            image = image.reshape((self.color_tex.getYSize(), self.color_tex.getXSize(), self.color_tex.getNumComponents()))
            image = np.flipud(image)
            image = image[..., :-1]  # remove alpha channel; if alpha values are needed, set alpha bits to 8
            images.append(image)

        if self.depth_tex:
            depth_data = self.depth_tex.getRamImage()
            if sys.version_info < (3, 0):
                depth_data = depth_data.get_data()
            depth_image = np.frombuffer(depth_data, np.float32)
            depth_image = depth_image.reshape((self.depth_tex.getYSize(), self.depth_tex.getXSize(), self.depth_tex.getNumComponents()))
            depth_image = np.flipud(depth_image)
            depth_image = (255 ** depth_image).astype(np.uint8)
            images.append(depth_image)
        return tuple(images)


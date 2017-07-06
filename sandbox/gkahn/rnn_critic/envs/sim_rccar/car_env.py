#!/usr/bin/env python
import time
import numpy as np
import sys
from sandbox.gkahn.rnn_critic.envs.sim_rccar.panda3d_camera_sensor import Panda3dCameraSensor
# from direct.showbase.DirectObject import DirectObject
from direct.showbase import DirectObject
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec3
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletVehicle
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import ZUp

import cv2
from rllab.spaces.box import Box

class CarEnv(DirectObject.DirectObject):

    def __init__(self, params={}):
        self._params = params
        self._use_vel = self._params.get('use_vel', True)
        if not self._params.get('visualize', False):
            loadPrcFileData('', 'window-type offscreen')

        # Defines base, render, loader
        try:
            ShowBase()
        except:
            pass # will enter here if multiple CarEnv created
        base.setBackgroundColor(0.1, 0.1, 0.8, 1)

        # World
        self._worldNP = render.attachNewNode('World')
        self._world = BulletWorld()
        self._world.setGravity(Vec3(0, 0, -9.81))
        self._dt = params.get('dt', 0.25)
        # TODO Light
#        alight = AmbientLight('ambientLight')
#        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
#        alightNP = render.attachNewNode(alight)
#
#        dlight = DirectionalLight('directionalLight')
#        dlight.setDirection(Vec3(1, 1, -1))
#        dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
#        dlightNP = render.attachNewNode(dlight)
#
#        render.clearLight()
#        render.setLight(alightNP)
#        render.setLight(dlightNP)

        # Camera
        self._camera_sensor = Panda3dCameraSensor(
            base,
            color=True,
            depth=True,
            size=(160,90))
        self._camera_node = self._camera_sensor.cam
        self._camera_node.setPos(0.0, 1.0, 1.0)
        self._camera_node.lookAt(0.0, 6.0, 0.0)
        
        self._back_camera_sensor = Panda3dCameraSensor(
            base,
            color=True,
            depth=True,
            size=(160,90))
        self._back_camera_node = self._back_camera_sensor.cam
        self._back_camera_node.setPos(0.0, -1.0, 1.0)
        self._back_camera_node.lookAt(0.0, -6.0, 0.0)

        # Vehicle
        shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
        ts = TransformState.makePos(Point3(0., 0., 0.5))
        self._vehicle_node = BulletRigidBodyNode('Vehicle')
        self._vehicle_node.addShape(shape, ts)
        self._mass = self._params.get('mass', 800.) 
        self._vehicle_node.setMass(self._mass)
        self._vehicle_node.setDeactivationEnabled(False)
        self._vehicle_node.setCcdSweptSphereRadius(1.0)
        self._vehicle_node.setCcdMotionThreshold(1e-7)
        self._vehicle_pointer = self._worldNP.attachNewNode(self._vehicle_node)

        self._camera_node.reparentTo(self._vehicle_pointer)
        self._back_camera_node.reparentTo(self._vehicle_pointer)

        self._world.attachRigidBody(self._vehicle_node)

        self._vehicle = BulletVehicle(self._world, self._vehicle_node)
        self._vehicle.setCoordinateSystem(ZUp)
        self._world.attachVehicle(self._vehicle)
        self._addWheel(Point3( 0.70,    1.05, 0.3), True)
        self._addWheel(Point3(-0.70,    1.05, 0.3), True)
        self._addWheel(Point3( 0.70, -1.05, 0.3), False)
        self._addWheel(Point3(-0.70, -1.05, 0.3), False)

        # Car Simulator
        self._setup()
        self._load_vehicle()

        # Input
        self.accept('escape', self._doExit)
        self.accept('r', self.reset)
        self.accept('f1', self._toggleWireframe)
        self.accept('f2', self._toggleTexture)
        self.accept('f5', self._doScreenshot)
        self.accept('q', self._forward_0)
        self.accept('w', self._forward_1)
        self.accept('e', self._forward_2)
        self.accept('a', self._left)
        self.accept('s', self._stop)
        self.accept('x', self._backward)
        self.accept('d', self._right)

        self._steering = 0.0       # degree
        self._engineForce = 0.0
        self._brakeForce = 0.0
        self._p = self._params.get('p', 2000.) 
        self._i = self._params.get('i', 0.)
        self._d = self._params.get('d', 90.)
        self._des_vel = None
        self._last_err = 0.0
        self._curr_time = 0.0
        self._accelClamp = self._params.get('accelClamp', 12.0)
        self._engineClamp = self._accelClamp * self._mass
        self._collision = False
#        taskMgr.add(self._update_task, 'updateWorld')
#        base.run()

        self._do_back_up = self._params.get('do_back_up', False)
        self._back_up = self._params.get('back_up', None)

        self.action_space = Box(low=np.array([-90., -8.]), high=np.array([90., 8.]))
        self.observation_space = Box(low=0, high=255, shape=(64, 36, 1)) # black and white

    # _____HANDLER_____

    def _doExit(self):
        sys.exit(1)

    def _toggleWireframe(self):
        base.toggleWireframe()

    def _toggleTexture(self):
        base.toggleTexture()

    def _doScreenshot(self):
        base.screenshot('Bullet')

    def _forward_0(self):
        self._des_vel = 14.4
        self._brakeForce = 0.0

    def _forward_1(self):
        self._des_vel = 28.8
        self._brakeForce = 0.0

    def _forward_2(self):
        self._des_vel = 48.
        self._brakeForce = 0.0

    def _stop(self):
        self._des_vel = 0.0
        self._brakeForce = 0.0

    def _backward(self):
        self._des_vel = -28.8
        self._brakeForce = 0.0

    def _right(self):
        self._steering = np.min([np.max([-15, self._steering - 5]), 0.0])

    def _left(self):
        self._steering = np.max([np.min([15, self._steering + 5]), 0.0])

    # Setup

    def _setup(self):
        if hasattr(self, '_model_path'):
            # Collidable objects
            visNP = loader.loadModel(self._model_path)
            visNP.clearModelNodes()
            visNP.reparentTo(render)
            pos = (0., 0., 0.)
            visNP.setPos(pos[0], pos[1], pos[2])

            bodyNPs = BulletHelper.fromCollisionSolids(visNP, True)
            for bodyNP in bodyNPs:
                bodyNP.reparentTo(render)
                bodyNP.setPos(pos[0], pos[1], pos[2])

                if isinstance(bodyNP.node(), BulletRigidBodyNode):
                    bodyNP.node().setMass(0.0)
                    bodyNP.node().setKinematic(True)
                    bodyNP.setCollideMask(BitMask32.allOn())
                    self._world.attachRigidBody(bodyNP.node())
        else:
            ground = self._worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
            ground.node().addShape(shape)
            ground.setCollideMask(BitMask32.allOn())

            self._world.attachRigidBody(ground.node())
    # Vehicle
 
    def _default_pos(self):
        return (0.0, 0.0, 0.0)

    def _default_hpr(self):
        return (0.0, 0.0, 0.0)

    def _update(self, dt=1.0):
        self._vehicle.setSteeringValue(self._steering, 0)
        self._vehicle.setSteeringValue(self._steering, 1)
        self._vehicle.setBrake(self._brakeForce, 2)
        self._vehicle.setBrake(self._brakeForce, 3)

        self._previous_pos = np.array(self._vehicle_pointer.getPos())
        self._previous_hpr = np.array(self._vehicle_pointer.getHpr())

        step = 0.05
        if dt > step:
            # TODO maybe change number of timesteps
            for i in range(int(dt/step)):
                if self._des_vel is not None:
                    vel = self._vehicle.getCurrentSpeedKmHour()
                    err = self._des_vel - vel
                    d_err = (err - self._last_err)/step
                    self._last_err = err
                    self._engineForce = np.clip(self._p * err + self._d * d_err, -self._engineClamp, self._engineClamp)
                self._vehicle.applyEngineForce(self._engineForce, 2)
                self._vehicle.applyEngineForce(self._engineForce, 3)
                self._world.doPhysics(step, 1, step)
                # Collision detection
                result = self._world.contactTest(self._vehicle_node)
                self._collision = result.getNumContacts() > 0
                if self._collision:
                    break
        else:
            self._curr_time += dt
            print('error')
            if self._curr_time > 0.05:
                if self._des_vel is not None:
                    vel = self._vehicle.getCurrentSpeedKmHour()
                    print(vel, self._curr_time)
                    err = self._des_vel - vel
                    d_err = (err - self._last_err)/0.05
                    self._last_err = err
                    self._engineForce = np.clip(self._p * err + self._d * d_err, -self._engineClamp, self._engineClamp)
                self._curr_time = 0.0
            self._vehicle.applyEngineForce(self._engineForce, 2)
            self._vehicle.applyEngineForce(self._engineForce, 3)
            self._world.doPhysics(dt, 1, dt)

            # Collision detection
            result = self._world.contactTest(self._vehicle_node)
            self._collision = result.getNumContacts() > 0
        
        if self._collision:
            self._load_vehicle(pos=self._previous_pos, hpr=self._previous_hpr)

    def _load_vehicle(self, pos=None, hpr=None):
        if pos is None:
            pos = self._default_pos()
        if hpr is None:
            hpr = self._default_hpr()
        self._steering = 0.0
        self._engineForce = 0.0
        
        self._vehicle.setSteeringValue(0.0, 0)
        self._vehicle.setSteeringValue(0.0, 1)
        self._vehicle.setBrake(1000.0, 2)
        self._vehicle.setBrake(1000.0, 3)
        self._vehicle.applyEngineForce(0.0, 2)
        self._vehicle.applyEngineForce(0.0, 3)

        self._vehicle_pointer.setPos(pos[0], pos[1], pos[2])
        if hpr is not None:
            self._vehicle_pointer.setHpr(hpr[0], hpr[1], hpr[2])
        
        while abs(self._vehicle.getCurrentSpeedKmHour()) > 4.0:
            self._world.doPhysics(self._dt, int(self._dt/0.05), 0.05)
            self._vehicle_pointer.setPos(pos[0], pos[1], pos[2])
            if hpr is not None:
                self._vehicle_pointer.setHpr(hpr[0], hpr[1], hpr[2])

    def _addWheel(self, pos, front):
        wheel = self._vehicle.createWheel()
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(0.25)
        wheel.setMaxSuspensionTravelCm(40.0)
        wheel.setSuspensionStiffness(40.0)
        wheel.setWheelsDampingRelaxation(2.3)
        wheel.setWheelsDampingCompression(4.4)
        wheel.setFrictionSlip(1e2)
        wheel.setRollInfluence(0.1)

    # Task

    def _update_task(self, task):
        dt = globalClock.getDt()
        self._update(dt=dt)
        return task.cont

    # Environment functions

    def _process_depth(self, image):
        mono_image = np.array(np.fromstring(image.tostring(), np.int32), np.float32)
        # TODO this is hardcoded                                                                                                                                                                                                    
        mono_image = (1.0653532e9 - mono_image) / (1.76e5) * 255
        im = cv2.resize(
            np.reshape(mono_image, (image.shape[0], image.shape[1])),
            self.observation_space.shape[:2],
            interpolation=cv2.INTER_AREA)
        return im.astype(np.uint8)

    def _get_observation(self):
        self._obs = self._camera_sensor.observe()
        self._back_obs = self._back_camera_sensor.observe()
        observation = {}
        observation['front_image'] = self._obs[0]
        observation['front_depth'] = self._obs[1]
        observation['back_image'] = self._back_obs[0]
        observation['back_depth'] = self._back_obs[1]
        observation['pos'] = np.array(self._vehicle_pointer.getPos())
        observation['hpr'] = np.array(self._vehicle_pointer.getHpr())
        # Convert from km/p to m/s
        observation['vel'] = self._vehicle.getCurrentSpeedKmHour() / 3.6
        observation['coll'] = self._collision 
        return self._process_depth(observation['front_depth'])

    def reset(self, pos=None, hpr=None):
        if self._do_back_up:
            steer = np.random.uniform(*self._back_up['cmd_steer'])
            vel = self._back_up['cmd_vel']
            u = [steer, vel]
            for _ in range(int(self._back_up['duration'] / self._dt)):
                self.step(u)
                if self._collision:
                    break
            for _ in range(int(1. / self._dt)):
                self.step([0., 0.])
        else:
            self._load_vehicle(pos=pos, hpr=hpr)
            result = self._world.contactTest(self._vehicle_node)
            self._collision = result.getNumContacts() > 0
        return self._get_observation()

    def step(self, action):
        self._steering = action[0]
        if action[1] == 0.0:
            self._brakeForce = 1000.
        else:
            self._brakeForce = 0.
        if self._use_vel:
            # Convert from m/s to km/h
            self._des_vel = action[1] * 3.6
        else:
            self._engineForce = self._engineClamp * \
                ((action[1] - 49.5) / 49.5)
        
        self._update(dt=self._dt)
        observation = self._get_observation()
        reward = action[1] - 10. * np.square(action[1]) * int(self._collision)
        done = self._collision
        info = {}
        return observation, reward, done, info

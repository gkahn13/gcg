import time
import numpy as np
import sys
from sandbox.gkahn.gcg.envs.rccar.panda3d_camera_sensor import Panda3dCameraSensor
from direct.showbase.DirectObject import DirectObject
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import BitMask32
from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletVehicle
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import ZUp

class CarEnv(DirectObject):
    def __init__(self, params={}):
        self._params = params
        if 'random_seed' in self._params:
            np.random.seed(self._params['random_seed'])
        self._use_vel = self._params.get('use_vel', True)
        self._run_as_task = self._params.get('run_as_task', False)
        self._do_back_up = self._params.get('do_back_up', False)
        self._use_depth = self._params.get('use_depth', False)
        self._use_back_cam = self._params.get('use_back_cam', False)
        self._collision_reward = self._params.get('collision_reward', 0.)
        if not self._params.get('visualize', False):
            loadPrcFileData('', 'window-type offscreen')

        # Defines base, render, loader

        try:
            ShowBase()
        except:
            pass
        
        base.setBackgroundColor(0.0, 0.0, 0.0, 1)

        # World
        self._worldNP = render.attachNewNode('World')
        self._world = BulletWorld()
        self._world.setGravity(Vec3(0, 0, -9.81))
        self._dt = params.get('dt', 0.25)
        self._step = 0.05

        # Vehicle
        shape = BulletBoxShape(Vec3(0.6, 1.0, 0.25))
        ts = TransformState.makePos(Point3(0., 0., 0.25))
        self._vehicle_node = BulletRigidBodyNode('Vehicle')
        self._vehicle_node.addShape(shape, ts)
        self._mass = self._params.get('mass', 10.) 
        self._vehicle_node.setMass(self._mass)
        self._vehicle_node.setDeactivationEnabled(False)
        self._vehicle_node.setCcdSweptSphereRadius(1.0)
        self._vehicle_node.setCcdMotionThreshold(1e-7)
        self._vehicle_pointer = self._worldNP.attachNewNode(self._vehicle_node)

        self._world.attachRigidBody(self._vehicle_node)

        self._vehicle = BulletVehicle(self._world, self._vehicle_node)
        self._vehicle.setCoordinateSystem(ZUp)
        self._world.attachVehicle(self._vehicle)
        self._addWheel(Point3( 0.3,  0.5, 0.07), True,  0.07)
        self._addWheel(Point3(-0.3,  0.5, 0.07), True,  0.07)
        self._addWheel(Point3( 0.3, -0.5, 0.07), False, 0.07)
        self._addWheel(Point3(-0.3, -0.5, 0.07), False, 0.07)

        # Camera
        size = self._params.get('size', [160, 90])
        hfov = self._params.get('hfov', 60)
        near_far = self._params.get('near_far', [0.1, 100.])
        self._camera_sensor = Panda3dCameraSensor(
            base,
            color=not self._use_depth,
            depth=self._use_depth,
            size=size,
            hfov=hfov,
            near_far=near_far,
            title='front cam')
        self._camera_node = self._camera_sensor.cam
        self._camera_node.setPos(0.0, 0.5, 0.375)
        self._camera_node.lookAt(0.0, 6.0, 0.0)
        self._camera_node.reparentTo(self._vehicle_pointer)

        if self._use_back_cam:
            self._back_camera_sensor = Panda3dCameraSensor(
                base,
                color=not self._use_depth,
                depth=self._use_depth,
                size=size,
                hfov=hfov,
                near_far=near_far,
                title='back cam')

            self._back_camera_node = self._back_camera_sensor.cam
            self._back_camera_node.setPos(0.0, -0.5, 0.375)
            self._back_camera_node.lookAt(0.0, -6.0, 0.0)
            self._back_camera_node.reparentTo(self._vehicle_pointer)
        
        # Car Simulator
        self._des_vel = None
        self._setup()
        
        # Input
        self.accept('escape', self._doExit)
        self.accept('r', self.reset)
        self.accept('f1', self._toggleWireframe)
        self.accept('f2', self._toggleTexture)
        self.accept('f3', self._view_image)
        self.accept('f5', self._doScreenshot)
        self.accept('q', self._forward_0)
        self.accept('w', self._forward_1)
        self.accept('e', self._forward_2)
        self.accept('a', self._left)
        self.accept('s', self._stop)
        self.accept('x', self._backward)
        self.accept('d', self._right)
        self.accept('m', self._mark)

        self._steering = 0.0       # degree
        self._engineForce = 0.0
        self._brakeForce = 0.0
        self._p = self._params.get('p', 1.25) 
        self._d = self._params.get('d', 0.0)
        self._last_err = 0.0
        self._curr_time = 0.0
        self._accelClamp = self._params.get('accelClamp', 2.0)
        self._engineClamp = self._accelClamp * self._mass
        self._collision = False
        if self._run_as_task:
            self._mark_d = 0.0
            taskMgr.add(self._update_task, 'updateWorld')
            base.run()

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
        self._des_vel = 1
        self._brakeForce = 0.0

    def _forward_1(self):
        self._des_vel = 2
        self._brakeForce = 0.0

    def _forward_2(self):
        self._des_vel = 4
        self._brakeForce = 0.0

    def _stop(self):
        self._des_vel = 0.0
        self._brakeForce = 0.0

    def _backward(self):
        self._des_vel = -4
        self._brakeForce = 0.0

    def _right(self):
        self._steering = np.min([np.max([-30, self._steering - 5]), 0.0])

    def _left(self):
        self._steering = np.max([np.min([30, self._steering + 5]), 0.0])

    def _view_image(self):
        from matplotlib import pyplot as plt
        image = self._camera_sensor.observe()[0]
        if self._use_depth:
            plt.imshow(image[:, :, 0], cmap='gray')
        else:
            import cv2
            def rgb2gray(rgb):
                return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

            image = rgb2gray(image)
            im = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)  # TODO how does this deal with aspect ratio
            plt.imshow(im.astype(np.uint8), cmap='Greys_r')
        plt.show()

    def _mark(self):
        self._mark_d = 0.0

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
            shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
            ground.node().addShape(shape)
            ground.setCollideMask(BitMask32.allOn())
            self._world.attachRigidBody(ground.node())
        self._place_vehicle()
        self._setup_light()
        self._setup_restart_pos()

    def _setup_restart_pos(self):
        self._restart_pos = []
        self._restart_index = 0
        if self._params.get('position_ranges', None) is not None:
            ranges = self._params['position_ranges']
            num_pos = self._params['num_pos']
            if self._params.get('range_type', 'random') == 'random':
                for _ in range(num_pos):
                    ran = ranges[np.random.randint(len(ranges))]
                    self._restart_pos.append(np.random.uniform(ran[0], ran[1]))
            elif self._params['range_type'] == 'fix_spacing':
                num_ran = len(ranges)
                num_per_ran = num_pos // num_ran
                for i in range(num_ran):
                    ran = ranges[i]
                    low = np.array(ran[0])
                    diff = np.array(ran[1]) - np.array(ran[0])
                    for j in range(num_per_ran):
                        val = diff * ((j + 0.0) / num_per_ran) + low
                        self._restart_pos.append(val)
        elif self._params.get('positions', None) is not None:
            self._restart_pos = self._params['positions']
        else:
            self._restart_pos = self._default_restart_pos()

    def _next_restart_pos_hpr(self):
        num = len(self._restart_pos)
        if num == 0:
            return None, None
        else:
            pos_hpr = self._restart_pos[self._restart_index]
            self._restart_index = (self._restart_index + 1) % num
            return pos_hpr[:3], pos_hpr[3:]

    def _next_random_restart_pos_hpr(self):
        num = len(self._restart_pos)
        if num == 0:
            return None, None
        else:
            index = np.random.randint(num)
            pos_hpr = self._restart_pos[index]
            self._restart_index = (self._restart_index + 1) % num
            return pos_hpr[:3], pos_hpr[3:]

    def _setup_light(self):
        alight = AmbientLight('ambientLight')
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        alightNP = render.attachNewNode(alight)
        render.clearLight()
        render.setLight(alightNP)

    # Vehicle
    def _default_pos(self):
        return (0.0, 0.0, 0.3)

    def _default_hpr(self):
        return (0.0, 0.0, 3.14)

    def _default_restart_pos():
        return [self._default_pos() + self._default_hpr()]

    def _get_speed(self):
        vel = self._vehicle.getCurrentSpeedKmHour() / 3.6
        return vel

    def _update(self, dt=1.0, coll_check=True):
        self._vehicle.setSteeringValue(self._steering, 0)
        self._vehicle.setSteeringValue(self._steering, 1)
        self._vehicle.setBrake(self._brakeForce, 0)
        self._vehicle.setBrake(self._brakeForce, 1)
        self._vehicle.setBrake(self._brakeForce, 2)
        self._vehicle.setBrake(self._brakeForce, 3)

        if dt >= self._step:
            # TODO maybe change number of timesteps
            for i in range(int(dt/self._step)):
                if self._des_vel is not None:
                    vel = self._get_speed()
                    err = self._des_vel - vel
                    d_err = (err - self._last_err) / self._step
                    self._last_err = err
                    self._engineForce = np.clip(self._p * err + self._d * d_err, -self._accelClamp, self._accelClamp) * self._mass
                self._vehicle.applyEngineForce(self._engineForce, 0)
                self._vehicle.applyEngineForce(self._engineForce, 1)
                self._vehicle.applyEngineForce(self._engineForce, 2)
                self._vehicle.applyEngineForce(self._engineForce, 3)
                self._world.doPhysics(self._step, 1, self._step)
            self._collision = self._is_contact()
        elif self._run_as_task:
            self._curr_time += dt
            if self._curr_time > 0.05:
                if self._des_vel is not None:
                    vel = self._get_speed()
                    self._mark_d += vel * self._curr_time
                    print(vel, self._mark_d, self._is_contact())
                    err = self._des_vel - vel
                    d_err = (err - self._last_err) / 0.05
                    self._last_err = err
                    self._engineForce = np.clip(self._p * err + self._d * d_err, -self._accelClamp, self._accelClamp) * self._mass
                self._curr_time = 0.0
                self._vehicle.applyEngineForce(self._engineForce, 0)
                self._vehicle.applyEngineForce(self._engineForce, 1)
                self._vehicle.applyEngineForce(self._engineForce, 2)
                self._vehicle.applyEngineForce(self._engineForce, 3)
            self._world.doPhysics(dt, 1, dt)
            self._collision = self._is_contact()
        else:
            raise ValueError("dt {0} s is too small for velocity control".format(dt))

    def _stop_car(self):
        self._steering = 0.0
        self._engineForce = 0.0
        self._vehicle.setSteeringValue(0.0, 0)
        self._vehicle.setSteeringValue(0.0, 1)
        self._vehicle.applyEngineForce(0.0, 0)
        self._vehicle.applyEngineForce(0.0, 1)
        self._vehicle.applyEngineForce(0.0, 2)
        self._vehicle.applyEngineForce(0.0, 3)
        
        if self._des_vel is not None:
            self._des_vel = 0
        
        self._vehicle_node.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        self._vehicle_node.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        for i in range(self._vehicle.getNumWheels()):
            wheel = self._vehicle.getWheel(i)
            wheel.setRotation(0.0)
        self._vehicle_node.clearForces()

    def _place_vehicle(self, pos=None, hpr=None):
        if pos is None:
            pos = self._default_pos()
        if hpr is None:
            hpr = self._default_hpr()
        self._vehicle_pointer.setPos(pos[0], pos[1], pos[2])
        self._vehicle_pointer.setHpr(hpr[0], hpr[1], hpr[2])
        self._stop_car()

    def _addWheel(self, pos, front, radius=0.25):
        wheel = self._vehicle.createWheel()
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(radius)
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
        self._get_observation()
        return task.cont

    # Helper functions

    def _get_observation(self):
        self._obs = self._camera_sensor.observe()
        observation = []
        observation.append(self._obs[0])
        if self._use_back_cam:
            self._back_obs = self._back_camera_sensor.observe()
            observation.append(self._back_obs[0])
        observation = np.concatenate(observation, axis=2)
        return observation

    def _get_reward(self):
        reward = self._collision_reward if self._collision else self._get_speed()
        return reward

    def _get_done(self):
        return self._collision
    
    def _get_info(self):
        info = {}
        info['pos'] = np.array(self._vehicle_pointer.getPos())
        info['hpr'] = np.array(self._vehicle_pointer.getHpr())
        info['vel'] = self._get_speed()
        info['coll'] = self._collision
        return info
    
    def _back_up(self):
        assert(self._use_vel)
        back_up_vel = self._params['back_up'].get('vel', -2.0) 
        self._des_vel = back_up_vel
        back_up_steer = self._params['back_up'].get('steer', (-5.0, 5.0))
        # TODO
        self._steering = np.random.uniform(*back_up_steer)
        self._brakeForce = 0.
        duration = self._params['back_up'].get('duration', 1.0)
        self._update(dt=duration)
        self._des_vel = 0.0
        self._steering = 0.0
        self._update(dt=duration)
        self._brakeForce = 0.

    def _is_contact(self):
        result = self._world.contactTest(self._vehicle_node)
        num_contacts = result.getNumContacts()
        return result.getNumContacts() > 0

    # Environment functions

    def reset(self, pos=None, hpr=None, hard_reset=False, random_reset=False):
        if self._do_back_up and not hard_reset and \
                pos is None and hpr is None:
            if self._collision:
                self._back_up()
        else:
            if pos is None and hpr is None:
                if random_reset:
                    pos, hpr = self._next_random_restart_pos_hpr()
                else:
                    pos, hpr = self._next_restart_pos_hpr()
            self._place_vehicle(pos=pos, hpr=hpr)
        self._collision = False
        return self._get_observation()

    def step(self, action):
        self._steering = action[0]
        if action[1] == 0.0:
            self._brakeForce = 1000.
        else:
            self._brakeForce = 0.
        if self._use_vel:
            # Convert from m/s to km/h
            self._des_vel = action[1]
        else:
            self._engineForce = self._engineClamp * \
                ((action[1] - 49.5) / 49.5)
        
        self._update(dt=self._dt)
        observation = self._get_observation()
        reward = self._get_reward() 
        done = self._get_done()
        info = self._get_info()
        return observation, reward, done, info

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True}
    env = CarEnv(params)

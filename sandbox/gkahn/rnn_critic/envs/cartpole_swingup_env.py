import numpy as np
import scipy.misc

from gym.envs.classic_control.cartpole import CartPoleEnv
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box

import pyglet
from pyglet import gl

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class CartPoleSwingupEnv(CartPoleEnv):
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 400

    def __init__(self, x_threshold=2.4, x_threshold_reward=-5000):
        CartPoleEnv.__init__(self)

        self.x_threshold = x_threshold
        self.x_threshold_reward = x_threshold_reward

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            1.1,
            1.1,
            np.finfo(np.float32).max])

        self.action_space = Discrete(2)
        self.observation_space = Box(-high, high)

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])

    def _step(self, action):
        x, x_dot, theta, theta_dot = self.state

        self.steps_beyond_done = None
        super(CartPoleSwingupEnv, self)._step(action)

        obs = self._get_obs()
        costs = np.power(angle_normalize(theta), 2) + \
                0.1 * np.power(theta_dot, 2)
        # reward = np.power(2*np.pi, 2) - \
        #          np.power(angle_normalize(theta), 2) - \
        #          0.1 * np.power(theta_dot, 2) - \
        #          0.001 * np.dot(action, action)
        done = (np.abs(x) > self.x_threshold)
        reward = -costs
        if done:
            reward = self.x_threshold_reward

        return obs, reward, done, {}

    def _reset(self):
        # x, x_dot, theta, theta_dot = state
        self.state = np.random.uniform([-0.05, -0.05, np.pi-0.05, -0.05],
                                       [0.05, 0.05, np.pi + 0.05, 0.05])
        self.steps_beyond_done = None
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = self.SCREEN_WIDTH
        screen_height = self.SCREEN_HEIGHT

        world_width = self.x_threshold*3
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0 * (self.x_threshold / 2.4)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            # self.track = rendering.Line((0,carty), (screen_width,carty))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class CartPoleSwingupImageEnv(CartPoleSwingupEnv):
    SCREEN_WIDTH = 150
    SCREEN_HEIGHT = 100

    def __init__(self, x_threshold=20, x_threshold_reward=-5000):
        CartPoleSwingupEnv.__init__(self, x_threshold=x_threshold, x_threshold_reward=x_threshold_reward)

        self.observation_space = Box(low=0, high=255, shape=(40, 60, 1)) # black and white

        self.reset()
        self.render()

    def _get_obs(self):
        im = self._render('rgb_array')
        im = scipy.misc.imresize(im, self.observation_space.shape[:2], interp='cubic')
        im = im[:, :, 0] * 0.299 + im[:, :, 1] * 0.587 + im[:, :, 2] * 0.114
        return im.astype(np.uint8).reshape(self.observation_space.shape)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = self.SCREEN_WIDTH
        screen_height = self.SCREEN_HEIGHT
        ratio = (screen_width / 600.)

        world_width = self.x_threshold * 3
        scale = screen_width / world_width
        carty = 100 * ratio # TOP OF CART
        polewidth = 10.0 * ratio
        polelen = scale * 1.2 * (self.x_threshold / 2.4)
        cartwidth = 50.0 * ratio
        cartheight = 30.0 * ratio

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            # self.track = rendering.Line((0,carty), (screen_width,carty))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        if mode == 'rgb_array':
            win = self.viewer.window
            win.clear()
            VP_W = screen_width
            VP_H = screen_height
            gl.glViewport(0, 0, VP_W, VP_H)
            for geom in self.viewer.geoms:
                geom.render()
            # import IPython; IPython.embed()

            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(screen_height, screen_width, 4)
            arr = arr[::-1, :, 0:3]
            return arr
        else:
            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

if __name__ == '__main__':
    env = CartPoleSwingupImageEnv(x_threshold=20)

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(1, 1)
    im = None
    import time

    start = time.time()
    for i in range(1000):
        obs, _, done, _ = env.step(1)
        # env.render()
        if im is None:
            im = ax.imshow(obs[:, :, 0], cmap='Greys_r')
            plt.show(block=False)
        else:
            im.set_array(obs[:, :, 0])
        f.canvas.draw()
        plt.pause(0.01)
        input(done)

    elapsed = time.time() - start
    print('FPS: {0}'.format(1000. / elapsed))
    import IPython; IPython.embed()

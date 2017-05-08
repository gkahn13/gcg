import numpy as np

from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete

from sandbox.gkahn.rnn_critic.envs.car import car_racing

STATE_W = 84
STATE_H = 84

class CollisionCarRacingFullEnv(car_racing.CarRacing):
    def __init__(self):
        super(CollisionCarRacingFullEnv, self).__init__()
        self.observation_space = Box(low=0, high=255, shape=(STATE_H, STATE_W, 1))
        self.repeat = 4

    def _convert_obs(self, obs):
        w_pad = int((car_racing.STATE_W - STATE_W) / 2.)
        h_pad = int((car_racing.STATE_H - STATE_H) / 2.)
        obs = obs[:-2 * h_pad, w_pad:-w_pad, :]
        obs = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        return obs.astype(np.uint8).reshape(self.observation_space.shape)

    def _reset(self):
        obs = super(CollisionCarRacingFullEnv, self)._reset()
        self._render()
        return obs

    def _step(self, action):
        reward = 0
        for i in range(self.repeat):
            obs_i, reward_i, _, _ = super(CollisionCarRacingFullEnv, self)._step(action)
            done = min([len(w.tiles) for w in self.car.wheels]) == 0
            reward += reward_i

            if done:
                break

        if done:
            reward = -100

        return self._convert_obs(obs_i), reward, done, {}

    @property
    def horizon(self):
        return 10000

    def play(self):
        from pyglet.window import key
        a = np.array([0.0, 0.0, 0.0])

        def key_press(k, mod):
            global restart
            if k == 0xff0d: restart = True
            if k == key.LEFT:  a[0] = -1.0
            if k == key.RIGHT: a[0] = +1.0
            if k == key.UP:    a[1] = +1.0
            if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

        def key_release(k, mod):
            if k == key.LEFT and a[0] == -1.0: a[0] = 0
            if k == key.RIGHT and a[0] == +1.0: a[0] = 0
            if k == key.UP:    a[1] = 0
            if k == key.DOWN:  a[2] = 0

        self.render()
        record_video = False
        if record_video:
            self.monitor.start('/tmp/video-test', force=True)
        self.viewer.window.on_key_press = key_press
        self.viewer.window.on_key_release = key_release
        while True:
            self.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                s, r, done, info = self.step(a)
                total_reward += r
                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                    # import matplotlib.pyplot as plt
                    # plt.imshow(s)
                    # plt.savefig("test.jpeg")
                steps += 1
                if not record_video:  # Faster, but you can as well call self.render() every time to play full window.
                    self.render()
                if done or restart: break
        self.close()

class CollisionCarRacingSteeringEnv(car_racing.CarRacing):
    def __init__(self):
        super(CollisionCarRacingSteeringEnv, self).__init__()
        self.action_space = Box(np.array([-1]), np.array([+1]))  # steer
        self.observation_space = Box(low=0, high=255, shape=(STATE_H, STATE_W, 1))
        self.repeat = 1

    def _convert_obs(self, obs):
        w_pad = int((car_racing.STATE_W - STATE_W) / 2.)
        h_pad = int((car_racing.STATE_H - STATE_H) / 2.)
        obs = obs[:-2 * h_pad, w_pad:-w_pad, :]
        obs = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        return obs.astype(np.uint8).reshape(self.observation_space.shape)

    def _reset(self):
        obs = super(CollisionCarRacingSteeringEnv, self)._reset()
        self._render()
        return obs

    def _step(self, action):
        if action is not None:
            action = [action[0], 0, 0]
            if self.t * car_racing.FPS < 30:
                action[1] = 1

        reward = 0
        for i in range(self.repeat):
            obs_i, _, _, _ = super(CollisionCarRacingSteeringEnv, self)._step(action)
            done = min([len(w.tiles) for w in self.car.wheels]) == 0

            if done:
                reward -= 100
                break
            else:
                reward += 1

        return self._convert_obs(obs_i), reward, done, {}

    @property
    def horizon(self):
        return 10000

    def play(self):
        from pyglet.window import key
        a = np.array([0.0, 0.0, 0.0])

        def key_press(k, mod):
            global restart
            if k == 0xff0d: restart = True
            if k == key.LEFT:  a[0] = -1.0
            if k == key.RIGHT: a[0] = +1.0
            if k == key.UP:    a[1] = +1.0
            if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

        def key_release(k, mod):
            if k == key.LEFT and a[0] == -1.0: a[0] = 0
            if k == key.RIGHT and a[0] == +1.0: a[0] = 0
            if k == key.UP:    a[1] = 0
            if k == key.DOWN:  a[2] = 0

        self.render()
        record_video = False
        if record_video:
            self.monitor.start('/tmp/video-test', force=True)
        self.viewer.window.on_key_press = key_press
        self.viewer.window.on_key_release = key_release
        while True:
            self.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                s, r, done, info = self.step([a[0]])
                total_reward += r
                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                    # import matplotlib.pyplot as plt
                    # plt.imshow(s)
                    # plt.savefig("test.jpeg")
                steps += 1
                if not record_video:  # Faster, but you can as well call self.render() every time to play full window.
                    self.render()
                if done or restart: break
        self.close()

class CollisionCarRacingDiscreteEnv(car_racing.CarRacing):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    GAS = 3
    BRAKE = 4

    def __init__(self):
        super(CollisionCarRacingDiscreteEnv, self).__init__()
        self.action_space = Discrete(5)
        self.observation_space = Box(low=0, high=255, shape=(STATE_H, STATE_W, 1))

    def _convert_obs(self, obs):
        w_pad = int((car_racing.STATE_W - STATE_W) / 2.)
        h_pad = int((car_racing.STATE_H - STATE_H) / 2.)
        obs = obs[:-2 * h_pad, w_pad:-w_pad, :]
        obs = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        return obs.astype(np.uint8).reshape(self.observation_space.shape)

    def _reset(self):
        obs = super(CollisionCarRacingDiscreteEnv, self)._reset()
        self._render()
        return obs

    def _step(self, action):
        if action == CollisionCarRacingDiscreteEnv.LEFT:
            action_cont = [-1, 0, 0]
        elif action == CollisionCarRacingDiscreteEnv.STRAIGHT:
            action_cont = [0, 0, 0]
        elif action == CollisionCarRacingDiscreteEnv.RIGHT:
            action_cont = [1, 0, 0]
        elif action == CollisionCarRacingDiscreteEnv.GAS:
            action_cont = [0, 1, 0]
        elif action == CollisionCarRacingDiscreteEnv.BRAKE:
            action_cont = [0, 0, 1]
        elif action is None:
            action_cont = None
        else:
            raise NotImplementedError
        obs, _, _, _ = super(CollisionCarRacingDiscreteEnv, self)._step(action_cont)

        done = min([len(w.tiles) for w in self.car.wheels]) == 0
        speed = np.linalg.norm(self.car.hull.linearVelocity)
        if done:
            reward = -5 * speed
        else:
            reward = 1e-1 * speed

        return self._convert_obs(obs), reward, done, {}

    @property
    def horizon(self):
        return 10000

    def play(self):
        from pyglet.window import key
        a = np.array([0])

        def key_press(k, mod):
            global restart
            if k == 0xff0d: restart = True
            elif k == key.LEFT:  a[0] = CollisionCarRacingDiscreteEnv.LEFT
            elif k == key.RIGHT: a[0] = CollisionCarRacingDiscreteEnv.RIGHT
            elif k == key.UP:    a[0] = CollisionCarRacingDiscreteEnv.GAS
            elif k == key.DOWN:  a[0] = CollisionCarRacingDiscreteEnv.BRAKE
            else: a[0] = CollisionCarRacingDiscreteEnv.STRAIGHT

        def key_release(k, mod):
            a[0] = CollisionCarRacingDiscreteEnv.STRAIGHT

        self.render()
        record_video = False
        if record_video:
            self.monitor.start('/tmp/video-test', force=True)
        self.viewer.window.on_key_press = key_press
        self.viewer.window.on_key_release = key_release
        while True:
            self.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True:
                s, r, done, info = self.step(a[0])
                total_reward += r
                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                    # import matplotlib.pyplot as plt
                    # plt.imshow(s)
                    # plt.savefig("test.jpeg")
                steps += 1
                if not record_video:  # Faster, but you can as well call self.render() every time to play full window.
                    self.render()
                if done or restart: break
        self.close()


if __name__ == "__main__":
    # import gym
    # import matplotlib.pyplot as plt
    #
    # env = CollisionCarRacingDiscreteEnv()
    # env.reset()
    #
    # f, ax = plt.subplots(1, 1)
    # im = None
    # import time
    #
    # start = time.time()
    # for _ in range(1000):
    #     obs, _, done, _ = env.step(CollisionCarRacingDiscreteEnv)
    #     if im is None:
    #         im = ax.imshow(obs[:, :, 0], cmap='Greys_r')
    #         plt.show(block=False)
    #     else:
    #         im.set_array(obs[:, :, 0])
    #     f.canvas.draw()
    #     plt.pause(0.01)
    #     input(done)
    # elapsed = time.time() - start
    # print('FPS: {0}'.format(1000. / elapsed))
    # import IPython; IPython.embed()

    env = CollisionCarRacingDiscreteEnv()
    env.play()

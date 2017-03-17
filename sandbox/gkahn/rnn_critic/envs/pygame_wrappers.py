from collections import deque
import numpy as np

import gym
from gym import spaces

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

        self.get_lives = None
        while True:
            if hasattr(env, 'game_state'):
                self.get_lives = env.game_state.lives
                break
            if not hasattr(env, 'env'):
                break
            env = env.env
        assert(self.get_lives is not None)

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.get_lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.get_lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class BlackAndWhiteWrapper(gym.Wrapper):
    def __init__(self, env=None):
        super(BlackAndWhiteWrapper, self).__init__(env)
        obs_shape = list(self.observation_space.shape)
        obs_shape[-1] = 1
        self.observation_space = spaces.Box(low=0, high=255, shape=tuple(obs_shape))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return BlackAndWhiteWrapper._process_black_and_white(obs), reward, done, info

    def _reset(self):
        return BlackAndWhiteWrapper._process_black_and_white(self.env.reset())

    @staticmethod
    def _process_black_and_white(img):
        img = img.astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        return img.astype(np.uint8)

class ClippedRewardsWrapper(gym.Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, np.sign(reward), done, info

def wrap_pygame(env):
    # make end-of-life == end-of-episode
    env = EpisodicLifeEnv(env)
    # max and skip env
    env = MaxAndSkipEnv(env)
    # make image black and white
    env = BlackAndWhiteWrapper(env)
    # clip rewards so just -1 0 +1
    env = ClippedRewardsWrapper(env)
    return env

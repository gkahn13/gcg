import time
import numpy as np
from pynput import keyboard

from sandbox.gkahn.rnn_critic.envs.rw_rccar.sensors_handler import SensorsHandler

from rllab.spaces.box import Box

class RWRCcarEnv:
    def __init__(self, params):
        self.action_space = Box(low=np.array([-45., -4.]), high=np.array([45., 4.]))
        self.observation_space = Box(low=0, high=255, shape=(36, 64, 1))

    def close(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def stop(self):
        pass

    @property
    def horizon(self):
        return int(1e5)

# class RWRCcarEnv:
#     def __init__(self, params):
#         self._sensors = SensorsHandler()
#         self._collision = False
#         self._params = params
#         self._do_back_up = params.get('do_back_up', True)
#         self._max_time_step = params.get('max_time_step', None)
#         self._time_step = 0
# #        self._next_time = None
#         self._last_obs_time = time.time() # TODO
#         self._button_end = False
#         self._listener = keyboard.Listener(on_press=self._on_press)
#         self._listener.start()
#
#         self.action_space = Box(low=np.array([-45., -4.]), high=np.array([45., 4.]))
#         self.observation_space = Box(low=0, high=255, shape=(36, 64, 1))
#
#     # Keyboard Listener function
#
#     def _on_press(self, key):
#         if str(key)[1] == 'p':
#             self.stop()
#             self._button_end = True
#
#     # Helper functions
#
# #    def _do_action(self, action, t, interrupt=False, vel_control=False):
# #        if self._next_time is not None and not interrupt:
# #            sleep_time = self._next_time - time.time()
# #            if sleep_time > 0:
# #                time.sleep(self._next_time - time.time())
# #                print(sleep_time)
# #        if not self._get_done():
# #            if vel_control:
# #                self._sensors.set_vel_cmd(action)
# #            else:
# #                self._sensors.set_motor_cmd(action)
# #            self._next_time = time.time() + t
#
#     def _do_action(self, action, t, absolute=False, vel_control=False):
#         if not self._get_done():
#             if vel_control:
#                 self._sensors.set_vel_cmd(action)
#             else:
#                 self._sensors.set_motor_cmd(action)
#             if absolute:
#                 sleep_time = t
#             else:
#                 sleep_time = (self._last_obs_time + t) - time.time()
#                 print(sleep_time)
#             if sleep_time > 0:
#                 time.sleep(sleep_time)
#
#     def stop(self):
#         self._do_action((0.0, 0.0), t=0.0, absolute=True, vel_control=False)
#
#     def _back_up(self):
#         back_up_vel = self._params['back_up'].get('vel', -2.0)
#         back_up_steer_ran = self._params['back_up'].get('steer', (-5.0, 5.0))
#         back_up_steer = np.random.uniform(*back_up_steer_ran)
#         duration = self._params['back_up'].get('duration', 1.0)
#         self._do_action((back_up_steer, back_up_vel), t=duration, absolute=True, vel_control=True)
#         self._do_action((0.0, 0.0), t=1.0, absolute=True, vel_control=False)
#
#     def _get_observation(self):
#         self._last_obs_time = time.time()
#         return self._sensors.get_image()
#
#     def _get_reward(self):
#         reward = -1.0 * int(self._sensors.get_crash())
#         return reward
#
#     def _get_done(self):
#         return self._sensors.get_crash() or self._button_end or self._time_step >= self._max_time_step
#
#     def _get_info(self):
#         info = {}
#         info['coll'] = self._sensors.get_crash()
#         info['flipped'] = self._sensors.get_flip()
#         info['steer'] = self._sensors.get_motor_data()[1]
#         info['motor'] = self._sensors.get_motor_data()[2]
#         info['vel'] = self._sensors.get_motor_data()[3]
#         info['acc'] = self._sensors.get_imu_data()[:3]
#         info['ori'] = self._sensors.get_imu_data()[3:]
#         return info
#
#     # Environment functions
#
#     def close(self):
#         self.stop()
#         self._sensors.close()
#         self._listener.stop()
#
#     def reset(self):
#         self._button_end = False
#         self._time_step = 0
#         if self._do_back_up:
#             if self._sensors.get_crash():
#                 self._sensors.reset_crash()
#                 self._back_up()
#         else:
#             self._do_action((0.0, 0.0), t=1.0, vel_control=False)
#         self._sensors.reset_crash()
#         return self._get_observation()
#
#     def step(self, action):
#         self._do_action(action, t=self._params['dt'], vel_control=self._params['use_vel'])
#         self._time_step += 1
#         observation = self._get_observation()
#         reward = self._get_reward()
#         done = self._get_done()
#         info = self._get_info()
#         if done:
#             self._do_action((0.0, 0.0), t=1.0, vel_control=False)
#         return observation, reward, done, info
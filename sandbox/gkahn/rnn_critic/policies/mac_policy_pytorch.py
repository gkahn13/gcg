from collections import defaultdict
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init

from rllab.core.serializable import Serializable
import rllab.misc.logger as logger

from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.gkahn.rnn_critic.utils import schedules
from sandbox.gkahn.rnn_critic.pytorch import torch_utils

from sandbox.gkahn.rnn_critic.utils.utils import timeit

class MACPolicyPytorch(Serializable):
    DEBUG = False # turn on/off debugging

    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        ### environment
        self._env_spec = kwargs.get('env_spec')

        ### model horizons
        self._N = kwargs.get('N') # number of returns to use (N-step)
        self._H = kwargs.get('H') # action planning horizon for training
        self._gamma = kwargs.get('gamma') # reward decay
        self._obs_history_len = kwargs.get('obs_history_len') # how many previous observations to use

        ### model architecture
        self._obs_hidden_layers = list(kwargs.get('obs_hidden_layers'))
        self._action_hidden_layers = list(kwargs.get('action_hidden_layers'))
        self._reward_hidden_layers = list(kwargs.get('reward_hidden_layers'))
        self._value_hidden_layers = list(kwargs.get('value_hidden_layers'))
        self._values_softmax_hidden_layers = list(kwargs.get('values_softmax_hidden_layers'))
        self._rnn_state_dim = kwargs.get('rnn_state_dim')
        self._use_lstm = kwargs.get('use_lstm')
        self._use_bilinear = kwargs.get('use_bilinear')
        self._activation = eval(kwargs.get('activation'))
        self._rnn_activation = eval(kwargs.get('rnn_activation'))
        self._use_conv = ('conv_hidden_layers' in kwargs) and ('conv_kernels' in kwargs) and \
                         ('conv_strides' in kwargs) and ('conv_activation' in kwargs)
        if self._use_conv:
            self._conv_hidden_layers = list(kwargs.get('conv_hidden_layers'))
            self._conv_kernels = list(kwargs.get('conv_kernels'))
            self._conv_strides = list(kwargs.get('conv_strides'))
            self._conv_activation = eval(kwargs.get('conv_activation'))

        ### target network
        self._use_target = kwargs.get('use_target')
        self._separate_target_params = kwargs.get('separate_target_params')

        ### how to combine n-step outputs
        self._values_softmax_policy = kwargs.get('values_softmax_policy')
        self._values_softmax_target = kwargs.get('values_softmax_target')

        ### training
        self._weight_decay = kwargs.get('weight_decay')
        self._lr_schedule = schedules.PiecewiseSchedule(**kwargs.get('lr_schedule'))
        self._grad_clip_norm = kwargs.get('grad_clip_norm')
        self._preprocess_params = kwargs.get('preprocess')
        self._gpu_device = kwargs.get('gpu_device', None)
        self._gpu_frac = kwargs.get('gpu_frac', None)

        ### action selection and exploration
        self._get_action_policy = kwargs.get('get_action_policy')
        self._get_action_target = kwargs.get('get_action_target')
        self._exploration_strategy = None # don't set in init b/c will then be Serialized
        self._num_exploration_strategy = 0  # keep track of how many times get action called

        ### setup the model
        self._policy_graph_get_action, self._graph_cost, self._opt = self._setup_graph()

        ### logging
        self._log_stats = defaultdict(list)

        assert((self._N == 1 and self._H == 1) or
               (self._N > 1 and self._H == 1) or
               (self._N > 1 and self._H > 1 and self._N == self._H))

    ##################
    ### Properties ###
    ##################

    @property
    def N(self):
        return self._N

    @property
    def _obs_is_im(self):
        return len(self._env_spec.observation_space.shape) > 1

    @property
    def _obs_shape(self):
        obs_shape = list(self._env_spec.observation_space.shape)
        if len(obs_shape) > 1:
            obs_shape[-1] = self._obs_history_len
            obs_shape = [obs_shape[2], obs_shape[0], obs_shape[1]]  # channels first
        return obs_shape

    @property
    def obs_history_len(self):
        return self._obs_history_len

    #############
    ### Graph ###
    #############

    class _GraphObservationsLowd(torch.nn.Module):
        def __init__(self, oself):
            super(MACPolicyPytorch._GraphObservationsLowd, self).__init__()
            self._oself = oself

            ### cnn
            if oself._use_conv:
                self._cnn = torch_utils.CNN(oself._obs_shape, oself._conv_hidden_layers,
                                            oself._conv_kernels, oself._conv_strides, oself._conv_activation)
                self._fc_input_dim = self._cnn.output_dim
            else:
                self._fc_input_dim = np.prod(oself._obs_shape)

            ### fc
            final_dim = 2 * oself._rnn_state_dim if oself._use_lstm else oself._rnn_state_dim
            self._fc = torch_utils.FullyConnected(self._fc_input_dim, oself._obs_hidden_layers + [final_dim],
                                                  oself._activation)

        def forward(self, observations):
            """
            :return: observations_lowd
            """
            if type(observations.data) == torch.cuda.ByteTensor:
                # is an image
                observations = (observations.float() / 255.) - 0.5

            observations_lowd = observations

            if self._oself._use_conv:
                observations_lowd = self._cnn(observations_lowd)
            observations_lowd = self._fc(observations_lowd)

            return observations_lowd

    class _GraphValues(torch.nn.Module):
        def __init__(self, oself, values_softmax):
            super(MACPolicyPytorch._GraphValues, self).__init__()

            self._oself = oself
            self._values_softmax = values_softmax

            action_space = oself._env_spec.action_space
            action_dim = action_space.flat_dim
            if isinstance(action_space, Discrete):
                self._action_fc = lambda x: x
            else:
                self._action_fc = torch_utils.FullyConnected(action_dim,
                                                             oself._action_hidden_layers + [oself._rnn_state_dim],
                                                             oself._activation)
            if oself._use_lstm:
                if isinstance(action_space, Discrete):
                    self._rnn = torch_utils.MuxLSTMCell(action_dim, oself._rnn_state_dim)
                else:
                    self._rnn = torch.nn.LSTMCell(oself._rnn_state_dim, oself._rnn_state_dim)
            else:
                if isinstance(action_space, Discrete):
                    self._rnn = torch_utils.MuxRNNCell(action_dim, oself._rnn_state_dim)
                else:
                    self._rnn = torch.nn.RNNCell(oself._rnn_state_dim, oself._rnn_state_dim)
            self._reward_fc = torch_utils.FullyConnected(oself._rnn_state_dim, oself._reward_hidden_layers + [1],
                                                         oself._activation)
            self._value_fc = torch_utils.FullyConnected(oself._rnn_state_dim, oself._reward_hidden_layers + [1],
                                                        oself._activation)
            if self._values_softmax == 'learned':
                self._values_softmax_fc = torch_utils.FullyConnected(oself._rnn_state_dim, oself._values_softmax_hidden_layers + [1],
                                                                     oself._activation)

        def forward(self, observations_lowd, actions):
            """
            :return: values, values_softmax
            """
            batch_size = actions.size(0)
            N = actions.size(1)

            ### rnn initial state
            if self._oself._use_lstm:
                h_t, c_t = observations_lowd.chunk(2, dim=1)
            else:
                h_t = observations_lowd

            ### rnn outputs
            outputs = []
            for action_t in actions.chunk(N, dim=1):
                input_t = self._action_fc(action_t[:, 0, :])
                # timeit.start('total')
                if self._oself._use_lstm:
                    h_t, c_t = self._rnn(input_t, (h_t, c_t))
                    outputs.append(c_t)
                else:
                    h_t = self._rnn(input_t, h_t)
                    outputs.append(h_t)
                # timeit.stop('total')

            ### rewards
            rewards = [self._reward_fc(output) for output in outputs]
            rewards = [rewards[-1]] + rewards[1:]  # shift right
            rewards[0][:] = 0  # zero out the first one

            ### nstep values
            nstep_values = [self._value_fc(output) for output in outputs]

            ### values
            values = []
            gammas = torch.autograd.Variable(
                torch.stack([torch.ones(batch_size).cuda() * np.power(self._oself._gamma, i) for i in range(N)],
                            dim=1))  # this takes all the time
            for n in range(N):
                returns = torch.cat(rewards[:n] + [nstep_values[n]], 1)
                values.append((gammas[:, :n + 1] * returns).sum(1))
            values = torch.cat(values, 1)

            ### values softmax
            if self._values_softmax == 'mean':
                values_softmax = torch.autograd.Variable(torch.ones(batch_size, N).cuda() / N)
            elif self._values_softmax == 'final':
                values_softmax = torch.autograd.Variable(torch.zeros(batch_size, N).cuda())
                values_softmax[:, -1] = 1
            elif self._values_softmax == 'learned':
                values_softmax = F.softmax(torch.cat([self._values_softmax_fc(output) for output in outputs], 1))
            else:
                raise NotImplementedError

            return values, values_softmax

    class _GraphGetAction(torch.nn.Module):
        def __init__(self, oself, graph_observations_lowd, graph_values, params):
            super(MACPolicyPytorch._GraphGetAction, self).__init__()

            self._oself = oself
            self._graph_observations_lowd = graph_observations_lowd
            self._graph_values = graph_values
            self._params = params

        @property
        def _H(self):
            return self._params['H']

        @property
        def _get_action_type(self):
            return self._params['type']

        def _sample_actions(self):
            action_space = self._oself._env_spec.action_space
            action_dim = action_space.flat_dim

            if self._get_action_type == 'random':
                K = self._params[self._get_action_type]['K']
                if isinstance(action_space, Discrete):
                    action_indices = torch.LongTensor(K * self._H, 1)
                    action_indices.random_(0, action_dim - 1)
                    actions = torch.FloatTensor(K * self._H, action_dim)
                    actions.zero_()
                    actions.scatter_(1, action_indices, 1)
                    actions.resize_(K, self._H, action_dim)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            return torch.autograd.Variable(actions.cuda())

        def forward(self, observations):
            """
            :return: get_action, get_action_value
            """
            action_dim = self._oself._env_spec.action_space.flat_dim

            ### observations lowd
            observations_lowd = self._graph_observations_lowd(observations)

            ### sample actions
            actions = self._sample_actions()

            ### repeat observations, tile actions if needed
            num_observations = observations.size(0)
            num_actions = actions.size(0)
            if num_observations != num_actions:
                observations_lowd_repeat = observations_lowd.repeat(1, num_actions).resize(
                    num_observations * num_actions,
                    observations_lowd.size(1))
                actions_tile = actions.repeat(num_observations, 1, 1)
            else:
                observations_lowd_repeat = observations_lowd
                actions_tile = actions

            ### do inference
            values, values_softmax = self._graph_values(observations_lowd_repeat, actions_tile)

            ### compute  best action and action value for each observation
            values = (values * values_softmax).sum(1)
            values = values.resize(num_observations, num_actions)
            get_action_value, get_action_indices = values.max(1)
            ### masking magic
            mask = torch_utils.one_hot((num_observations, num_actions), get_action_indices) # [num_observations, num_actions]
            mask = mask.resize(num_observations * num_actions) # [num_observations * num_actions]
            mask = torch_utils.expand_dims(mask, 1).repeat(1, action_dim) # [num_observations * num_actions, action_dim]
            get_action = (actions_tile[:, 0, :] * mask.float()).resize(num_observations, num_actions, action_dim).sum(1).squeeze(1)
            if self._oself.DEBUG:
                assert(np.all(get_action.data.cpu().numpy() ==
                              actions[:, 0, :][get_action_indices.squeeze(1).data].data.cpu().numpy()))

            return get_action, get_action_value

    class _GraphCost(torch.nn.Module):
        def __init__(self, oself,
                     policy_graph_observations_lowd, policy_graph_values,
                     target_graph_observations_lowd, target_graph_values, target_action_params):
            super(MACPolicyPytorch._GraphCost, self).__init__()

            self._oself = oself
            self._policy_graph_observations_lowd = policy_graph_observations_lowd
            self._policy_graph_values = policy_graph_values
            self._target_graph_observations_lowd = target_graph_observations_lowd
            self._target_graph_values = target_graph_values
            self._target_get_action = oself._GraphGetAction(oself,
                                                            target_graph_observations_lowd, target_graph_values,
                                                            target_action_params)

        def policy_parameters(self):
            return itertools.chain(self._policy_graph_observations_lowd.parameters(),
                                   self._policy_graph_values.parameters())

        def target_parameters(self):
            return itertools.chain(self._target_graph_observations_lowd.parameters(),
                                   self._target_graph_values.parameters())

        def update_target(self):
            for policy_param, target_param in zip(self.policy_parameters(), self.target_parameters()):
                target_param.data = policy_param.data

        def forward(self, observations, actions, dones, rewards, target_observations):
            """
            :return: cost
            """
            batch_size = rewards.size(0)
            N = rewards.size(1) - 1

            ### compute rewards
            gammas = torch.autograd.Variable(
                torch.stack([torch.ones(batch_size).cuda() * np.power(self._oself._gamma, i) for i in range(N + 1)], dim=1))
            discounted_sum_rewards = (rewards * gammas).cumsum(1)
            ### compute target values
            if self._oself._use_target:
                target_observations_stacked = torch.cat([target_observations[:, h:h + self._oself._obs_history_len, :]
                                                         for h in range(N)], 0)
                target_values = self._target_get_action(target_observations_stacked)[1].resize(batch_size, N).detach()
            else:
                target_values = None
            ### compute policy values
            observations_lowd = self._policy_graph_observations_lowd(observations)
            policy_values, policy_values_softmax = self._policy_graph_values(observations_lowd, actions)

            ### form and return bellman error
            cost = 0
            for n in range(N):
                error = discounted_sum_rewards[:, n + 1] - policy_values[:, n]
                if target_values is not None:
                    error += (1 - dones[:, n]).float() * (target_values[:, n] * np.power(self._oself._gamma, n + 1))
                cost += (1 / float(N)) * (policy_values_softmax[:, n] * error * error).mean()

            return cost

    def _setup_graph(self):
        ### policy
        policy_graph_observations_lowd = self._GraphObservationsLowd(self)
        policy_graph_values = self._GraphValues(self, self._values_softmax_policy)
        policy_graph_get_action = self._GraphGetAction(self, policy_graph_observations_lowd, policy_graph_values,
                                                       params=self._get_action_policy)

        ### target
        if self._separate_target_params:
            target_graph_observations_lowd = self._GraphObservationsLowd(self)
            target_graph_values = self._GraphValues(self, self._values_softmax_target)
        else:
            target_graph_observations_lowd = policy_graph_observations_lowd
            target_graph_values = policy_graph_values

        ### cost
        graph_cost = self._GraphCost(self, policy_graph_observations_lowd, policy_graph_values,
                                     target_graph_observations_lowd, target_graph_values,
                                     self._get_action_target)
        opt = torch.optim.Adam(graph_cost.policy_parameters())

        ### cuda-fy
        policy_graph_get_action.cuda()
        graph_cost.cuda()

        return policy_graph_get_action, graph_cost, opt

    ################
    ### Training ###
    ################

    def update_preprocess(self, preprocess_stats):
        pass

    def update_target(self):
        if self.DEBUG:
            diff_before = [np.linalg.norm(p.data.cpu().numpy() - t.data.cpu().numpy())
                           for p, t in zip(self._graph_cost.policy_parameters(), self._graph_cost.target_parameters())]

        self._graph_cost.update_target()

        if self.DEBUG:
            diff_after = [np.linalg.norm(p.data.cpu().numpy() - t.data.cpu().numpy())
                         for p, t in zip(self._graph_cost.policy_parameters(), self._graph_cost.target_parameters())]
            assert(max(diff_after) <= 1e-5)

    def train_step(self, step, observations, actions, rewards, dones, use_target):
        self._opt.zero_grad()
        cost = self._graph_cost(torch.autograd.Variable(torch.from_numpy(observations[:, :self._obs_history_len, :]).cuda()),
                                torch.autograd.Variable(torch.from_numpy(actions[:, :self._N, :]).cuda()),
                                torch.autograd.Variable(torch.from_numpy(np.logical_or(not use_target, dones).astype(int)).cuda()),
                                torch.autograd.Variable(torch.from_numpy(rewards).cuda()),
                                torch.autograd.Variable(torch.from_numpy(observations[:, 1:, :]).cuda()))
        cost.backward()

        if self.DEBUG:
            cost_params_before = [p.data.cpu().numpy() for p in self._graph_cost.parameters()]
            target_params_before = [p.data.cpu().numpy() for p in self._graph_cost._target_get_action.parameters()]

        self._opt.step()

        if self.DEBUG:
            cost_params_after = [p.data.cpu().numpy() for p in self._graph_cost.parameters()]
            target_params_after = [p.data.cpu().numpy() for p in self._graph_cost._target_get_action.parameters()]

            cost_params_diff = [np.linalg.norm(b - a) for (b, a) in zip(cost_params_before, cost_params_after)]
            target_params_diff = [np.linalg.norm(b - a) for (b, a) in zip(target_params_before, target_params_after)]
            assert(max(target_params_diff) < 1e-5)

        self._log_stats['Cost'].append(cost.cpu().data[0])

    ######################
    ### Policy methods ###
    ######################

    def set_exploration_strategy(self, exploration_strategy):
        self._exploration_strategy = exploration_strategy
        self._num_exploration_strategy = 0 # reset

    def get_action(self, observation):
        chosen_actions, action_info = self.get_actions([observation])
        return chosen_actions[0], action_info

    def get_actions(self, observations):
        torch_observations = torch.autograd.Variable(torch.from_numpy(np.array(observations)).cuda())
        actions = self._policy_graph_get_action(torch_observations)[0].cpu().data.numpy()

        chosen_actions = []
        for i, (observation_i, action_i) in enumerate(zip(observations, actions)):
            if isinstance(self._env_spec.action_space, Discrete):
                action_i = int(action_i.argmax())
            if self._exploration_strategy is not None:
                exploration_func = lambda: None
                exploration_func.get_action = lambda _: (action_i, dict())
                action_i = self._exploration_strategy.get_action(self._num_exploration_strategy,
                                                                 observation_i,
                                                                 exploration_func)
                self._num_exploration_strategy += 1

            chosen_actions.append(action_i)

        return chosen_actions, {}

    @property
    def recurrent(self):
        return False

    def terminate(self):
        pass

    ######################
    ### Saving/loading ###
    ######################

    def state_dicts(self):
        return self._graph_cost.state_dict(), self._opt.state_dict()

    def load_state_dicts(self, graph_cost_state_dict, opt_state_dict):
        self._graph_cost.load_state_dict(graph_cost_state_dict)
        self._opt.load_state_dict(opt_state_dict)

    ###############
    ### Logging ###
    ###############

    def log(self):
        for k in sorted(self._log_stats.keys()):
            if k == 'Depth':
                logger.record_tabular(k + 'Mean', np.mean(self._log_stats[k]))
                logger.record_tabular(k + 'Std', np.std(self._log_stats[k]))
            else:
                logger.record_tabular(k, np.mean(self._log_stats[k]))
        self._log_stats.clear()

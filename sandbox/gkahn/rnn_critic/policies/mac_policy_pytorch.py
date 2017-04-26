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

from sandbox.gkahn.rnn_critic.utils.utils import timeit

def numpy_to_variable(x):
    return torch.autograd.Variable(torch.from_numpy(x).cuda())

class FullyConnected(torch.nn.Module):
    def __init__(self, input_dim, layer_dims, activation, output_activation=None):
        super(FullyConnected, self).__init__()

        ### create layers
        self._layers = []
        for output_dim in layer_dims:
            self._layers.append(torch.nn.Linear(input_dim, output_dim).cuda())
            input_dim = output_dim

        ### activations
        self._activations = [activation] * (len(layer_dims) - 1) + [output_activation]

        ### initialize weights
        for layer in self._layers:
            torch.nn.init.xavier_uniform(layer.weight, gain=np.sqrt(2.))
            torch.nn.init.normal(layer.bias, std=0.1)

    def forward(self, x):
        for i, (layer, activation) in enumerate(zip(self._layers, self._activations)):
            x = layer(x)
            if activation is not None:
                x = activation(x)

        return x

class CNN(torch.nn.Module):
    def __init__(self, input_shape, channel_dims, kernels, strides, activation):
        """
        :param input_shape: [channels, H, W]
        """
        super(CNN, self).__init__()
        self._input_shape = input_shape

        ### create layers
        self._layers = []
        input_channel = input_shape[0]
        for output_channel, kernel, stride in zip(channel_dims, kernels, strides):
            self._layers.append(torch.nn.Conv2d(input_channel, output_channel, kernel, stride).cuda())
            input_channel = output_channel

        ### activations
        self._activation = activation

        ### initialize weights
        for layer in self._layers:
            torch.nn.init.xavier_uniform(layer.weight, gain=np.sqrt(2.))
            torch.nn.init.constant(layer.bias, 0.1)

        ### calculate output dim
        tmp_forward = torch.nn.Sequential(*self._layers)(torch.autograd.Variable(torch.ones(1, *input_shape).cuda()))
        self._output_dim = int(np.prod(tmp_forward.size()[1:]))

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        """
        :param x: [H, W, channels]
        """
        x = x.resize(x.size(0), *self._input_shape)

        ### conv
        for layer in self._layers:
            x = self._activation(layer(x))

        ### flatten
        x = x.view(-1, self.output_dim)

        return x

def tile(x, num, axis=0):
    return torch.cat([x] * num, axis)

def repeat(x, num, axis=0): # TODO this is super slow
    return torch.stack(list(itertools.chain(*[[v] * num for v in torch.unbind(x, dim=axis)])), dim=axis)

class MACObservationsLowd(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MACObservationsLowd, self).__init__()

        self._setup_variables(**kwargs)
        self._setup_modules()

    #############
    ### Setup ###
    #############

    def _setup_variables(self, **kwargs):
        ### environment
        self._env_spec = kwargs.get('env_spec')

        ###  model horizons
        self._obs_history_len = kwargs.get('obs_history_len')  # how many previous observations to use

        ### model
        self._obs_hidden_layers = list(kwargs.get('obs_hidden_layers'))
        self._activation = eval(kwargs.get('activation'))
        self._use_conv = ('conv_hidden_layers' in kwargs) and ('conv_kernels' in kwargs) and \
                         ('conv_strides' in kwargs) and ('conv_activation' in kwargs)
        if self._use_conv:
            self._conv_hidden_layers = list(kwargs.get('conv_hidden_layers'))
            self._conv_kernels = list(kwargs.get('conv_kernels'))
            self._conv_strides = list(kwargs.get('conv_strides'))
            self._conv_activation = eval(kwargs.get('conv_activation'))
        self._rnn_state_dim = kwargs.get('rnn_state_dim')
        self._use_lstm = kwargs.get('use_lstm')

    def _setup_modules(self):
        ### cnn
        if self._use_conv:
            self._cnn = CNN(self._obs_shape, self._conv_hidden_layers,
                            self._conv_kernels, self._conv_strides, self._conv_activation)
            self._fc_input_dim = self._cnn.output_dim
        else:
            self._fc_input_dim = np.prod(self._obs_shape)

        ### fc
        final_dim = 2 * self._rnn_state_dim if self._use_lstm else self._rnn_state_dim
        self._fc = FullyConnected(self._fc_input_dim, self._obs_hidden_layers + [final_dim], self._activation)

    ##################
    ### Properties ###
    ##################

    @property
    def _obs_shape(self):
        obs_shape = list(self._env_spec.observation_space.shape)
        if len(obs_shape) > 1:
            obs_shape[-1] = self._obs_history_len
            obs_shape = [obs_shape[2], obs_shape[0], obs_shape[1]] # channels first
        return obs_shape

    ###############
    ### Forward ###
    ###############

    def forward(self, observations):
        """
        :return: observations_lowd
        """
        observations_lowd = observations

        if self._use_conv:
            observations_lowd = self._cnn(observations_lowd)
        observations_lowd = self._fc(observations_lowd)

        return observations_lowd

class MACValues(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MACValues, self).__init__()

        self._setup_variables(**kwargs)
        self._setup_modules()

    #############
    ### Setup ###
    #############

    def _setup_variables(self, **kwargs):
        ### environment
        self._env_spec = kwargs.get('env_spec')

        ### model horizons
        self._gamma = kwargs.get('gamma')  # reward decay

        ### model architecture
        # rnn
        self._rnn_state_dim = kwargs.get('rnn_state_dim')
        self._use_lstm = kwargs.get('use_lstm')
        # outputs
        self._action_hidden_layers = list(kwargs.get('action_hidden_layers'))
        self._reward_hidden_layers = list(kwargs.get('reward_hidden_layers'))
        self._value_hidden_layers = list(kwargs.get('value_hidden_layers'))
        self._lambda_hidden_layers = list(kwargs.get('lambda_hidden_layers'))
        self._activation = eval(kwargs.get('activation'))

    def _setup_modules(self):
        assert(self._use_lstm) # TODO temp

        self._action_fc = FullyConnected(self._env_spec.action_space.flat_dim,
                                         self._action_hidden_layers + [self._rnn_state_dim],
                                         self._activation)
        self._rnn = torch.nn.LSTMCell(self._rnn_state_dim, self._rnn_state_dim)
        # self._rnn = torch.nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
        #                           num_layers=config.n_layers)
        self._reward_fc = FullyConnected(self._rnn_state_dim, self._reward_hidden_layers + [1], self._activation)
        self._value_fc = FullyConnected(self._rnn_state_dim, self._reward_hidden_layers + [1], self._activation)

    ###############
    ### Forward ###
    ###############

    def forward(self, observations_lowd, actions):
        """
        :return: values, values_softmax
        """
        # timeit.start('total')
        batch_size = actions.size(0)
        N = actions.size(1)

        ### rnn initial state
        if self._use_lstm:
            h_t, c_t = observations_lowd.chunk(2, dim=1)
        else:
            pass

        ### rnn outputs
        outputs = []
        for action_t in actions.chunk(N, dim=1):
            # timeit.start('MacValues:actions')
            input_t = self._action_fc(action_t[:, 0, :])
            # timeit.stop('MacValues:actions')
            # timeit.start('MacValues:RNN')
            h_t, c_t = self._rnn(input_t, (h_t, c_t))
            # timeit.stop('MacValues:RNN')
            outputs.append(c_t)

            # outputs.append(torch.autograd.Variable(torch.zeros(batch_size, self._rnn_state_dim).cuda()))

        ### rewards
        # timeit.start('MacValues:rewards')
        rewards = [self._reward_fc(output) for output in outputs]
        rewards = [rewards[-1]] + rewards[1:] # shift right
        rewards[0][:] = 0 # zero out the first one
        # timeit.stop('MacValues:rewards')

        ### nstep values
        # timeit.start('MacValues:nstep_values')
        nstep_values = [self._value_fc(output) for output in outputs]
        # timeit.stop('MacValues:nstep_values')

        ### values
        # timeit.start('MacValues:values')
        values = []
        gammas = torch.autograd.Variable(
            torch.stack([torch.ones(batch_size).cuda() * np.power(self._gamma, i) for i in range(N)], dim=1)) # TODO: this takes all the time
        for n in range(N):
            returns = torch.cat(rewards[:n] + [nstep_values[n]], 1)
            values.append((gammas[:, :n+1] * returns).sum(1))
        values = torch.cat(values, 1)
        # timeit.stop('MacValues:values')

        ### values softmax
        values_softmax = torch.autograd.Variable(torch.ones(batch_size, N).cuda() / N) # TODO hard coded

        # timeit.stop('total')

        return values, values_softmax

class MACGetAction(torch.nn.Module):
    def __init__(self, mac_observations_lowd, mac_values, params, **kwargs):
        super(MACGetAction, self).__init__()

        self._mac_observations_lowd = mac_observations_lowd
        self._mac_values = mac_values
        self._params = params

        self._setup_variables(**kwargs)
        self._setup_modules()

    #############
    ### Setup ###
    #############

    def _setup_variables(self, **kwargs):
        ### environment
        self._env_spec = kwargs.get('env_spec')

    def _setup_modules(self):
        pass

    ##################
    ### Properties ###
    ##################

    @property
    def _H(self):
        return self._params['H']

    @property
    def _get_action_type(self):
        return self._params['type']

    ###############
    ### Forward ###
    ###############

    def _sample_actions(self):
        action_dim = self._env_spec.action_space.flat_dim

        if self._get_action_type == 'random':
            K = self._params[self._get_action_type]['K']
            if isinstance(self._env_spec.action_space, Discrete):
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
        ### observations lowd # TODO: negligible time
        observations_lowd = self._mac_observations_lowd(observations)

        ### sample actions # TODO: negligible time
        actions = self._sample_actions()

        ### repeat observations, tile actions if needed # TODO: negligible time
        num_observations = observations.size(0)
        num_actions = actions.size(0)
        if num_observations != num_actions:
            observations_lowd_repeat = observations_lowd.repeat(1, num_actions).resize(num_observations * num_actions,
                                                                                       observations_lowd.size(1))
            actions_tile = actions.repeat(num_observations, 1, 1)
        else:
            observations_lowd_repeat = observations_lowd
            actions_tile = actions

        ### do inference # TODO: all time spent here
        values, values_softmax = self._mac_values(observations_lowd_repeat, actions_tile)

        ### compute  best action and action value for each observation
        values = (values * values_softmax).sum(1)
        values = values.resize(num_observations, num_actions)
        get_action_value, get_action_indices = values.max(1)
        get_action = actions[:, 0, :][get_action_indices.squeeze(1).data]

        return get_action, get_action_value

class MACCost(torch.nn.Module):
    def __init__(self, policy_mac_observations_lowd, policy_mac_values,
                       target_mac_observations_lowd, target_mac_values, target_action_params,
                       **kwargs):
        super(MACCost, self).__init__()

        self._policy_mac_observations_lowd = policy_mac_observations_lowd
        self._policy_mac_values = policy_mac_values
        self._target_get_action = MACGetAction(target_mac_observations_lowd, target_mac_values,  target_action_params,
                                               **kwargs)

        self._setup_variables(**kwargs)
        self._setup_modules()

    #############
    ### Setup ###
    #############

    def _setup_variables(self, **kwargs):
        ### environment
        self._gamma = kwargs.get('gamma')

        ### model horizons
        self._obs_history_len = kwargs.get('obs_history_len')  # how many previous observations to use

    def _setup_modules(self):
        pass

    ###############
    ### Forward ###
    ###############

    def forward(self, observations, actions, dones, rewards, target_observations):
        """
        :return: cost
        """
        # timeit.start('MacCost')

        batch_size = rewards.size(0)
        N = rewards.size(1) - 1

        ### compute rewards
        # timeit.start('MacCost:Rewards')
        gammas = torch.autograd.Variable(
            torch.stack([torch.ones(batch_size).cuda() * np.power(self._gamma, i) for i in range(N+1)], dim=1))
        discounted_sum_rewards = (rewards * gammas).cumsum(1)
        # timeit.stop('MacCost:Rewards')
        ### compute target values
        # timeit.start('MacCost:TargetValues')
        target_observations_stacked = torch.cat([target_observations[:, h:h+self._obs_history_len, :]
                                                 for h in range(N)], 0)
        target_values = self._target_get_action(target_observations_stacked)[1].resize(batch_size, N).detach()
        # timeit.stop('MacCost:TargetValues')
        ### compute policy values
        # timeit.start('MacCost:PolicyValues')
        observations_lowd = self._policy_mac_observations_lowd(observations)
        policy_values, policy_values_softmax = self._policy_mac_values(observations_lowd, actions)
        # timeit.stop('MacCost:PolicyValues')

        ### form and return bellman error
        # timeit.start('MacCost:Bellman')
        cost = 0
        for n in range(N):
            error = discounted_sum_rewards[:, n+1] \
                    + (1 - dones[:, n]).float() * (target_values[:, n] * np.power(self._gamma, n+1)) \
                    - policy_values[:, n]
            cost += (1 / float(N)) * (policy_values_softmax[:, n] * error * error).mean()
        # timeit.stop('MacCost:Bellman')

        # timeit.stop('MacCost')

        return cost

class MACPolicyPytorch(Serializable):
    def __init__(self, **kwargs):
        ### policy
        policy_mac_observations_lowd = MACObservationsLowd(**kwargs)
        policy_mac_values = MACValues(**kwargs)
        self._policy_mac_get_action = MACGetAction(policy_mac_observations_lowd, policy_mac_values,
                                                   params=kwargs.get('get_action_test'),
                                                   **kwargs)

        ### target
        target_mac_observations_lowd = MACObservationsLowd(**kwargs)
        target_mac_values = MACValues(**kwargs)

        ### cost
        self._mac_cost = MACCost(policy_mac_observations_lowd, policy_mac_values,
                                 target_mac_observations_lowd, target_mac_values, kwargs.get('get_action_target'),
                                 **kwargs)
        self._optimizer = torch.optim.Adam(self._mac_cost.parameters())

        ### cuda-fy
        self._policy_mac_get_action.cuda()
        self._mac_cost.cuda()

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
        self._lambda_hidden_layers = list(kwargs.get('lambda_hidden_layers'))
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
        self._values_softmax = kwargs.get('values_softmax') # which value horizons to train over
        self._use_target = kwargs.get('use_target')
        self._separate_target_params = kwargs.get('separate_target_params')

        ### training
        self._weight_decay = kwargs.get('weight_decay')
        self._lr_schedule = schedules.PiecewiseSchedule(**kwargs.get('lr_schedule'))
        self._grad_clip_norm = kwargs.get('grad_clip_norm')
        self._preprocess_params = kwargs.get('preprocess')
        self._gpu_device = kwargs.get('gpu_device', None)
        self._gpu_frac = kwargs.get('gpu_frac', None)

        ### action selection and exploration
        self._get_action_test = kwargs.get('get_action_test')
        self._get_action_target = kwargs.get('get_action_target')
        self._exploration_strategy = None # don't set in init b/c will then be Serialized
        self._num_exploration_strategy = 0  # keep track of how many times get action called

        ### setup the model
        # self._tf_debug = dict()
        # self._tf_dict = self._graph_setup()

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
        if self._obs_is_im:
            obs_shape[0] = self._obs_history_len
        return obs_shape

    @property
    def obs_history_len(self):
        return self._obs_history_len

    ########################
    ### graph operations ###
    ########################

    class _GraphObsToObsLowd(torch.nn.Module):
        def __init__(self, obs_shape, hidden_layers, activation,
                     conv_hidden_layers=None, conv_kernels=None, conv_strides=None, conv_activation=None):
            ### convolutions / input setup
            self._use_conv = (conv_hidden_layers is not None and conv_kernels is not None and
                              conv_strides is not None and conv_activation is not None)
            if self._use_conv:
                self._conv_layers = []
                in_channel = obs_shape[0]
                for out_channel, kernel, stride in zip(conv_hidden_layers, conv_kernels, conv_strides):
                    conv_layer = torch.nn.Conv2d(in_channel, out_channel, kernel, stride)
                    torch.nn.init.xavier_uniform(conv_layer.weight, gain=np.sqrt(2.))
                    torch.nn.init.constant(conv_layer.bias, 0.1)
                    self._conv_layers.append(conv_layer)
                    in_channel = out_channel

                ### compute flatten size by running graph and checking output
                tmp_conv_output = torch.nn.Sequential(self._conv_layers)(torch.autograd.Variable(torch.ones(1, *obs_shape)))
                self._flatten_size = int(np.prod(tmp_conv_output.size()[1:]))
            else:
                self._conv_layers = []
                self._flatten_size = np.prod(obs_shape)
            self._conv_activation = conv_activation

            ### fully connected
            self._fc_layers = []
            in_size = self._flatten_size
            for out_size in hidden_layers:
                fc_layer = torch.nn.Linear(in_size, out_size)
                torch.nn.init.xavier_uniform(fc_layer.weight, gain=np.sqrt(2.))
                torch.nn.init.normal(fc_layer.bias, std=0.1)
                self._fc_layers.append(fc_layer)
                in_size = out_size
            self._activation = activation

            super(MACPolicyPytorch._GraphObsToLowd, self).__init__()

        def forward(self, observations):
            x = observations

            ### convolutions
            if self._use_conv:
                # TODO: convert to float, subtract mean divide by 255
                for conv_layer in self._conv_layers:
                    x = self._conv_activation(conv_layer(x))
            ### flatten
            x = x.view(-1, self._flatten_size)
            ### fully connected
            for i, fc_layer in enumerate(self._fc_layers):
                x = fc_layer(x)
                if i < len(self._fc_layers) - 1:
                    x = self._activation(x)

            return x

    class _GraphObsLowdToValues(torch.nn.Module):
        def __init__(self, action_dim, rnn_state_dim, use_lstm):
            self._action_dim = action_dim
            self._rnn_state_dim = rnn_state_dim
            self._use_lstm = use_lstm

            assert(self._use_lstm) # TODO: tmp

            if self._use_lstm:
                self._rnn = torch.nn.LSTMCell(self._rnn_state_dim, self._rnn_state_dim)
            else:
                pass

            super(MACPolicyPytorch._GraphInference, self).__init__()

        def forward(self, obs_lowd, actions):
            pass

    def _graph_setup(self):
        ### policy
        policy_obslowd = self._GraphObsToObsLowd(obs_shape=self._obs_shape,
                                                 hidden_layers=self._obs_hidden_layers +
                                                               [2*self._rnn_state_dim if self._use_lstm else
                                                                self._rnn_state_dim],
                                                 activation=self._activation,
                                                 conv_hidden_layers=getattr(self, 'conv_hidden_layers', None),
                                                 conv_kernels=getattr(self, 'conv_kernels', None),
                                                 conv_strides=getattr(self, 'conv_strides', None),
                                                 conv_activation=getattr(self, 'conv_activation', None))
        policy_rnn_outputs = self._GraphObsLowdToRNNOutputs(action_dim=self._env_spec.action_space.flat_dim,
                                                            rnn_state_dim=self._rnn_state_dim,
                                                            use_lstm=self._use_lstm)
        policy_values = None
        policy_values_softmax = None
        policy_get_action = None

        ### target

        cost = None

        return policy_get_action, cost

    ################
    ### Training ###
    ################

    def update_preprocess(self, preprocess_stats):
        pass

    def update_target(self):
        pass

    def train_step(self, step, observations, actions, rewards, dones, use_target):
        self._optimizer.zero_grad()
        observations = observations.astype(np.float32)
        # timeit.start('TrainStep:forward')
        cost = self._mac_cost(torch.autograd.Variable(torch.from_numpy(observations[:, :self._obs_history_len, :]).cuda().float()),
                              torch.autograd.Variable(torch.from_numpy(actions[:, :self._N, :]).cuda()),
                              torch.autograd.Variable(torch.from_numpy(dones.astype(int)).cuda()),
                              torch.autograd.Variable(torch.from_numpy(rewards).cuda()),
                              torch.autograd.Variable(torch.from_numpy(observations[:, 1:, :]).cuda().float()))
        # timeit.stop('TrainStep:forward')
        # timeit.start('TrainStep:backward')
        cost.backward()
        # timeit.stop('TrainStep:backward')
        # timeit.start('TrainStep:opt')
        self._optimizer.step()
        # timeit.stop('TrainStep:opt')

        # print('train_step'); import IPython; IPython.embed()

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
        torch_observations = torch.autograd.Variable(torch.from_numpy(np.array(observations)).cuda().float())
        actions = self._policy_mac_get_action(torch_observations)[0].cpu().data.numpy()

        # print('get_actions'); import IPython; IPython.embed()

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

    def get_params_internal(self, **tags):
        pass

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

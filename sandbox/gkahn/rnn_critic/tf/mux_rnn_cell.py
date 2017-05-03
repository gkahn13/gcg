import tensorflow as tf

### inner python imports
from tensorflow.python.ops.rnn_cell import RNNCell

class BasicMuxRNNCell(RNNCell):
    def __init__(self, num_actions, num_units, activation=None):
        """
        Each cell consists of num_actions BasicRNNCells, each with num_units

        :param num_actions: assumes actions are one-hot vectors
        """
        activation = activation if activation is not None else tf.tanh
        self._rnn_cells = [tf.nn.rnn_cell.BasicRNNCell(num_units, activation=activation) for _ in range(num_actions)]
        self._num_actions = num_actions
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        Call each rnn_cell with the current state (and zero inputs)
        Use inputs to mux with rnn_cell is outputted
        Inputs are one hot, use it to mux the outputs by masking
        """
        if scope is None:
            scope = ''
        hs, states = zip(*[rnn_cell(tf.zeros(tf.shape(state), dtype=state.dtype), state, scope=scope+'mux_{0}'.format(i))
                         for i, rnn_cell in enumerate(self._rnn_cells)])
        masks = tf.split(split_dim=1, num_split=self._num_actions, value=inputs)
        new_h = tf.add_n([mask * h for mask, h in zip(masks, hs)])
        new_state = tf.add_n([mask * state for mask, state in zip(masks, states)])

        return new_h, new_state

class BasicMuxLSTMCell(RNNCell):
    def __init__(self, num_actions, num_units, state_is_tuple=True, activation=None,
                 initializer=None):
        """
        Each cell consists of num_actions BasicRNNCells, each with num_units

        :param num_actions: assumes actions are one-hot vectors
        """
        activation = activation if activation is not None else tf.tanh
        initializer = initializer if initializer is not None else tf.contrib.layers.xavier_initializer()
        assert(state_is_tuple)
        self._rnn_cells = [tf.nn.rnn_cell.LSTMCell(num_units,
                                                   state_is_tuple=state_is_tuple,
                                                   activation=activation,
                                                   initializer=initializer) for _ in range(num_actions)]

        self._num_actions = num_actions
        self._num_units = num_units
        self._activation = activation

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """
        Call each rnn_cell with the current state (and zero inputs)
        Use inputs to mux with rnn_cell is outputted
        Inputs are one hot, use it to mux the outputs by masking
        ASSUMES state_is_tuple
        """
        if scope is None:
            scope = ''
        hs, states = zip(*[rnn_cell(tf.zeros(tf.shape(state[0]), dtype=state.dtype), state, scope=scope+'mux_{0}'.format(i))
                         for i, rnn_cell in enumerate(self._rnn_cells)])
        masks = tf.split(split_dim=1, num_split=self._num_actions, value=inputs)
        new_h = tf.add_n([mask * h for mask, h in zip(masks, hs)])
        new_state0 = tf.add_n([mask * state[0] for mask, state in zip(masks, states)])
        new_state1 = tf.add_n([mask * state[1] for mask, state in zip(masks, states)])
        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_state0, new_state1)

        return new_h, new_state

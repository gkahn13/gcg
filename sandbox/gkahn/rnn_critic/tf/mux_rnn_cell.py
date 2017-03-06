import tensorflow as tf

### inner python imports
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

class BasicMuxRNNCell(RNNCell):
    """The most basic RNN cell."""

    def __init__(self, num_actions, num_units, activation=tf.tanh, reuse=None):
        self._rnn_cells = [tf.nn.rnn_cell.BasicRNNCell(num_units,
                                                       activation=activation,
                                                       reuse=reuse) for _ in range(num_actions)]

        self._num_actions = num_actions
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

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
        outputs = [rnn_cell(tf.zeros(tf.shape(state), dtype=state.dtype), state, scope=scope)
                   for rnn_cell in self._rnn_cells]
        masks = tf.split(split_dim=1, num_split=self._num_actions, value=inputs)
        output = tf.add_n([mask * output for mask, output in zip(masks, outputs)])

        return output, output


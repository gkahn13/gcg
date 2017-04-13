import tensorflow as tf


def linear(args, output_size, bias, bias_start=0.0, use_l2_loss=False, scope=None):
    """
    Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    # assert args #was causing error in upgraded tensorflow
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    if use_l2_loss:
        l_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
    else:
        l_regularizer = None

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size],
                                 initializer=tf.uniform_unit_scaling_initializer(), regularizer=l_regularizer)

        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)

        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size],
                                    initializer=tf.constant_initializer(bias_start), regularizer=l_regularizer)

    return res + bias_term


def multiplicative_integration(list_of_inputs, output_size, initial_bias_value=0.0,
                               weights_already_calculated=False, use_l2_loss=False, scope=None):
    '''
    expects len(2) for list of inputs and will perform integrative multiplication
    weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
    '''
    with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
        if len(list_of_inputs) != 2: raise ValueError('list of inputs must be 2, you have:', len(list_of_inputs))

        if weights_already_calculated:  # if you already have weights you want to insert from batch norm
            Wx = list_of_inputs[0]
            Uz = list_of_inputs[1]

        else:
            with tf.variable_scope('Calculate_Wx_mulint'):
                Wx = linear(list_of_inputs[0], output_size, False, use_l2_loss=use_l2_loss)
            with tf.variable_scope("Calculate_Uz_mulint"):
                Uz = linear(list_of_inputs[1], output_size, False, use_l2_loss=use_l2_loss)

        with tf.variable_scope("multiplicative_integration"):
            alpha = tf.get_variable('mulint_alpha', [output_size],
                                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.1))

            beta1, beta2 = tf.split(0, 2,
                                    tf.get_variable('mulint_params_betas', [output_size * 2],
                                                    initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.1)))

            original_bias = tf.get_variable('mulint_original_bias', [output_size],
                                            initializer=tf.truncated_normal_initializer(mean=initial_bias_value,
                                                                                        stddev=0.1))

        final_output = alpha * Wx * Uz + beta1 * Uz + beta2 * Wx + original_bias

    return final_output


class BasicMulintRNNCell(tf.nn.rnn_cell.BasicRNNCell):
    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            output = self._activation(multiplicative_integration([inputs, state], self._num_units))
        return output, output


class BasicMulintLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(1, 2, state)
            concat = multiplicative_integration([inputs, h], 4 * self._num_units, 0.0)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(1, 4, concat)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])
            return new_h, new_state

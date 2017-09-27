import tensorflow as tf

##################
### Operations ###
##################

def linear(args, output_size, dtype=tf.float32, scope=None):
    with tf.variable_scope(scope or "linear"):
        if isinstance(args, list) or isinstance(args, tuple):
            if len(args) != 1:
                inputs = tf.concat(args, axis=1)
            else:
                inputs = args[0]
        else:
            inputs = args
            args = [args]
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            else:
                total_arg_size += shape[1].value
        dtype = args[0].dtype
        weights = tf.get_variable(
            "weights",
            [total_arg_size, output_size],
            dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer(dtype=dtype))
        output = tf.matmul(inputs, weights)
    return output

def multiplicative_integration(
            list_of_inputs,
            output_size,
            initial_bias_value=0.0,
            weights_already_calculated=False,
            reg_collection=None,
            dtype=tf.float32,
            scope=None):
    '''
    expects len(2) for list of inputs and will perform integrative multiplication
    weights_already_calculated will treat the list of inputs as Wx and Uz and is useful for batch normed inputs
    '''
    with tf.variable_scope(scope or 'double_inputs_multiple_integration'):
        if len(list_of_inputs) != 2: raise ValueError('list of inputs must be 2, you have:', len(list_of_inputs))

        # TODO can do batch norm in FC
        if weights_already_calculated:  # if you already have weights you want to insert from batch norm
            Wx = list_of_inputs[0]
            Uz = list_of_inputs[1]

        else:
            Wx = linear(
                list_of_inputs[0],
                output_size,
                dtype=dtype,
                reg_collection=reg_collection,
                scope="Calculate_Wx_mulint")

            Uz = linear(
                list_of_inputs[1],
                output_size,
                dtype=dtype,
                reg_collection=reg_collection,
                scope="Calculate_Uz_mulint")

        with tf.variable_scope("multiplicative_integration"):
            alpha = tf.get_variable(
                'mulint_alpha',
                [output_size],
                dtype=dtype,
                initializer=tf.truncated_normal_initializer(
                    mean=1.0,
                    stddev=0.1,
                    dtype=dtype))

            beta1, beta2 = tf.split(
                tf.get_variable(
                    'mulint_params_betas',
                    [output_size * 2],
                    dtype=dtype,
                    initializer=tf.truncated_normal_initializer(
                        mean=0.5,
                        stddev=0.1,
                        dtype=dtype)),
                2,
                axis=0)

            original_bias = tf.get_variable(
                'mulint_original_bias',
                [output_size],
                dtype=dtype,
                initializer=tf.truncated_normal_initializer(
                    mean=initial_bias_value,
                    stddev=0.1,
                    dtype=dtype))

        final_output = alpha * Wx * Uz + beta1 * Uz + beta2 * Wx + original_bias

    return final_output

def layer_norm(
        inputs,
        center=True,
        scale=True,
        reuse=None,
        trainable=True,
        epsilon=1e-4,
        scope=None):
    # TODO
    # Assumes that inputs is 2D
    # add to collections in order to do l2 norm
    with tf.variable_scope(
            scope,
            default_name='LayerNorm',
            reuse=reuse):
        shape = tf.shape(inputs)
        param_shape = (inputs.get_shape()[1],)
        dtype = inputs.dtype.base_dtype
        beta = tf.zeros((shape[0],))
        gamma = tf.ones((shape[0],))
#        beta = tf.get_variable(
#            'beta',
#            shape=param_shape,
#            dtype=dtype,
#            initializer=tf.zeros_initializer(),
#            trainable=trainable and center)
#        gamma = tf.get_variable(
#            'gamma',
#            shape=param_shape,
#            dtype=dtype,
#            initializer=tf.ones_initializer(),
#            trainable=trainable and scale)
        inputs_T = tf.transpose(inputs)
        inputs_T_reshaped = tf.reshape(inputs_T, (shape[1], shape[0], 1, 1))
        outputs_T_reshaped, _, _ = tf.nn.fused_batch_norm(
            inputs_T_reshaped,
            scale=gamma,
            offset=beta,
            is_training=True,
            epsilon=epsilon,
            data_format='NCHW')
        outputs_reshaped = tf.transpose(outputs_T_reshaped, (1, 0, 2, 3))
        outputs = tf.reshape(outputs_reshaped, shape)
        return outputs

#############
### Cells ###
#############

class DpRNNCell(tf.nn.rnn_cell.BasicRNNCell):
    def __init__(
            self,
            num_units,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            weights_scope=None):
        self._num_units = num_units
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype

        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights = tf.get_variable(
                "weights",
                [num_inputs + num_units, num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B). With same dropout at every time step."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"

            ins = tf.concat([inputs, state], axis=1)
            output = self._activation(tf.matmul(ins, self._weights))

            if self._dropout_mask is not None:
                output = output * self._dropout_mask

        return output, output


class DpMulintRNNCell(DpRNNCell):
    def __init__(
            self,
            num_units,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            use_layer_norm=False,
            weights_scope=None):

        self._num_units = num_units
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        self._use_layer_norm = use_layer_norm

        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights_W = tf.get_variable(
                "weights_W",
                [num_inputs, num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

            self._weights_U = tf.get_variable(
                "weights_U",
                [num_units, num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            Wx = tf.matmul(inputs, self._weights_W)
            Uz = tf.matmul(state, self._weights_U)
            if self._use_layer_norm:
                Wx = tf.contrib.layers.layer_norm(
                    Wx,
                    center=False,
                    scale=False)
                Uz = tf.contrib.layers.layer_norm(
                    Uz,
                    center=False,
                    scale=False)
            output = self._activation(
                multiplicative_integration(
                    [Wx, Uz],
                    self._num_units,
                    dtype=self._dtype,
                    weights_already_calculated=True))

            if self._dropout_mask is not None:
                output = output * self._dropout_mask

        return output, output


class DpLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(
            self,
            num_units,
            forget_bias=1.0,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            weights_scope=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        self._state_is_tuple = True

        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights = tf.get_variable(
                "weights",
                [num_inputs + num_units, 4 * num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic LSTM with same dropout at every time step."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"

            c, h = state
            ins = tf.concat([inputs, h], axis=1)
            output = self._activation(tf.matmul(ins, self._weights))

            i, j, f, o = tf.split(output, 4, axis=1)

            forget = c * tf.nn.sigmoid(f + self._forget_bias)
            new = tf.nn.sigmoid(i) * self._activation(j)
            new_c = forget + new

            # TODO make sure this is correct
            if self._dropout_mask is not None:
                new_c = new_c * self._dropout_mask

            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        return new_h, new_state


class DpMulintLSTMCell(DpLSTMCell):
    def __init__(
            self,
            num_units,
            forget_bias=1.0,
            dropout_mask=None,
            activation=tf.tanh,
            dtype=tf.float32,
            num_inputs=None,
            use_layer_norm=False,
            weights_scope=None):

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._dropout_mask = dropout_mask
        self._activation = activation
        self._dtype = dtype
        self._use_layer_norm = use_layer_norm
        self._state_is_tuple = True

        with tf.variable_scope(weights_scope or type(self).__name__):
            self._weights_W = tf.get_variable(
                "weights_W",
                [num_inputs, 4 * num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

            self._weights_U = tf.get_variable(
                "weights_U",
                [num_units, 4 * num_units],
                dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                regularizer=tf.contrib.layers.l2_regularizer(0.5))

    def __call__(
            self,
            inputs,
            state,
            scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"

            c, h = state

            Wx = tf.matmul(inputs, self._weights_W)
            Uz = tf.matmul(h, self._weights_U)
            if self._use_layer_norm:
                Wx = tf.contrib.layers.layer_norm(
                    Wx,
                    center=False,
                    scale=False)
                Uz = tf.contrib.layers.layer_norm(
                    Uz,
                    center=False,
                    scale=False)
            output = self._activation(
                multiplicative_integration(
                    [Wx, Uz],
                    4 * self._num_units,
                    dtype=self._dtype,
                    weights_already_calculated=True))

            i, j, f, o = tf.split(output, 4, axis=1)

            forget = c * tf.nn.sigmoid(f + self._forget_bias)
            new = tf.nn.sigmoid(i) * self._activation(j)
            new_c = forget + new

            #            if self._use_layer_norm:
            #                new_c = tf.contrib.layers.layer_norm(
            #                    new_c,
            #                    center=True,
            #                    scale=True)

            # TODO make sure this is correct
            if self._dropout_mask is not None:
                new_c = new_c * self._dropout_mask

            if self._use_layer_norm:
                norm_c = tf.contrib.layers.layer_norm(
                    new_c,
                    center=True,
                    scale=True)
                new_h = self._activation(norm_c) * tf.nn.sigmoid(o)
            else:
                new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state

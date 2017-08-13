import tensorflow as tf

from sandbox.gkahn.rnn_critic.tf import rnn_cell

def convnn(inputs,
           params,
           scope='convnn',
           dtype=tf.float32,
           data_format='NHWC',
           reuse=False,
           is_training=True):

    if params['conv_activation'] == 'relu':
        conv_activation = tf.nn.relu
    else:
        raise NotImplementedError(
            'Conv activation {0} is not valid'.format(
                params['conv_activation']))

    if 'output_activation' not in params:
        output_activation = None
    elif params['output_activation'] == 'sigmoid':
        output_activation = tf.nn.sigmoid
    elif params['output_activation'] == 'softmax':
        output_activation = tf.nn.softmax
    elif params['output_activation'] == 'spatial_softmax':
        output_activation = lambda x: spatial_soft_argmax(x, dtype)
    elif params['output_activation'] == 'tanh':
        output_activation = tf.nn.tanh
    elif params['output_activation'] == 'relu':
        output_activation = tf.nn.relu
    else:
        raise NotImplementedError(
            'Output activation {0} is not valid'.format(
                params['output_activation']))

    kernels = params['kernels']
    filters = params['filters']
    strides = params['strides']
    # Assuming all paddings will be the same type
    padding = params.get('padding', 'SAME')
    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i in range(len(kernels)):
            if i == len(kernels) - 1:
                activation = output_activation
            else:
                activation = conv_activation
            if params.get('use_batch_norm', False):
                normalizer_fn = tf.contrib.layers.batch_norm
                scale = not (activation == tf.nn.relu or activation is None)
                normalizer_params = {
                    'is_training': is_training,
                    'data_format': data_format,
                    'fused': True,
                    'decay': params.get('batch_norm_decay', 0.999),
                    # 'zero_debias_moving_mean': True,
                    'scale': scale
                }
            else:
                normalizer_fn = None
                normalizer_params = None
            next_layer_input = tf.contrib.layers.conv2d(
                inputs=next_layer_input,
                num_outputs=filters[i],
                data_format=data_format,
                kernel_size=kernels[i],
                stride=strides[i],
                padding=padding,
                activation_fn=activation,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                trainable=True)

    output = next_layer_input
    return output


def fcnn(inputs,
         params,
         dp_masks=None,
         num_dp=1,
         dtype=tf.float32,
         data_format='NCHW',
         scope='fcnn',
         reuse=False,
         is_training=True):

    if params['hidden_activation'] == 'relu':
        hidden_activation = tf.nn.relu
    elif params['hidden_activation'] == 'tanh':
        hidden_activation = tf.nn.tanh
    else:
        raise NotImplementedError(
            'Hidden activation {0} is not valid'.format(
                params['hidden_activation']))

    if 'output_activation' not in params or params['output_activation'] == 'None':
        output_activation = None
    elif params['output_activation'] == 'sigmoid':
        output_activation = tf.nn.sigmoid
    elif params['output_activation'] == 'softmax':
        output_activation = tf.nn.softmax
    elif params['output_activation'] == 'relu':
        output_activation = tf.nn.relu
    elif params['output_activation'] == 'tanh':
        output_activation = tf.nn.tanh
    else:
        raise NotImplementedError(
            'Output activation {0} is not valid'.format(
                params['output_activation']))

    hidden_layers = params.get('hidden_layers', [])
    output_dim = params['output_dim']
    dropout = params.get('dropout', None)
    if dp_masks is not None or dropout is None:
        dp_return_masks = None
    else:
        dp_return_masks = []
        distribution = tf.contrib.distributions.Uniform()

    dims = hidden_layers + [output_dim]

    next_layer_input = inputs
    with tf.variable_scope(scope, reuse=reuse):
        for i, dim in enumerate(dims):
            if i == len(dims) - 1:
                activation = output_activation
            else:
                activation = hidden_activation
            if params.get('use_batch_norm', False):
                normalizer_fn = tf.contrib.layers.batch_norm
                scale = not (activation == tf.nn.relu or activation is None)
                normalizer_params = {
                    'is_training': is_training,
                    'data_format': data_format,
                    'fused': True,
                    'decay': params.get('batch_norm_decay', 0.999),
                    'zero_debias_moving_mean': True,
                    'scale': scale
                }
            else:
                normalizer_fn = None
                normalizer_params = None
            next_layer_input = tf.contrib.layers.fully_connected(
                inputs=next_layer_input,
                num_outputs=dim,
                activation_fn=activation,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                weights_initializer=tf.contrib.layers.xavier_initializer(dtype=dtype),
                biases_initializer=tf.constant_initializer(0., dtype=dtype),
                weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                trainable=True)

            if dropout is not None:
                assert (type(dropout) is float and 0 < dropout and dropout <= 1.0)
                if dp_masks is not None:
                    next_layer_input = next_layer_input * dp_masks[i]
                else:
                    # Shape is not well defined without reshaping
                    shape = tf.shape(next_layer_input)
                    if num_dp > 1:
                        sample = distribution.sample(shape[0] / num_dp, dim)
                        sample = tf.concat([sample] * num_dp, axis=0)
                    else:
                        sample = distribution.sample(shape)
                    sample = tf.reshape(sample, (-1, dim))
                    mask = tf.cast(sample < dropout, dtype) / dropout
                    next_layer_input = next_layer_input * mask
                    dp_return_masks.append(mask)

        output = next_layer_input

    return output, dp_return_masks


def rnn(inputs,
        params,
        initial_state=None,
        dp_masks=None,
        num_dp=1,
        dtype=tf.float32,
        scope='rnn',
        reuse=False):
    """
    inputs is shape [batch_size x T x features].
    """
    num_cells = params['num_cells']
    if params['cell_type'] == 'rnn':
        cell_type = rnn_cell.DpRNNCell
        if initial_state is not None:
            initial_state = tf.split(initial_state, num_cells, axis=1)
            num_units = initial_state[0].get_shape()[1].value
    elif params['cell_type'] == 'mulint_rnn':
        cell_type = rnn_cell.DpMulintRNNCell
        if initial_state is not None:
            initial_state = tuple(tf.split(initial_state, num_cells, axis=1))
            num_units = initial_state[0].get_shape()[1].value
    elif params['cell_type'] == 'lstm':
        cell_type = rnn_cell.DpLSTMCell
        if initial_state is not None:
            states = tf.split(initial_state, 2 * num_cells, axis=1)
            num_units = states[0].get_shape()[1].value
            initial_state = []
            for i in range(num_cells):
                initial_state.append(tf.nn.rnn_cell.LSTMStateTuple(states[i * 2], states[i * 2 + 1]))
            initial_state = tuple(initial_state)
    elif params['cell_type'] == 'mulint_lstm':
        cell_type = rnn_cell.DpMulintLSTMCell
        if initial_state is not None:
            states = tf.split(initial_state, 2 * num_cells, axis=1)
            num_units = states[0].get_shape()[1].value
            initial_state = []
            for i in range(num_cells):
                initial_state.append(tf.nn.rnn_cell.LSTMStateTuple(states[i * 2], states[i * 2 + 1]))
            initial_state = tuple(initial_state)
    else:
        raise NotImplementedError(
            'Cell type {0} is not valid'.format(params['cell_type']))

    if initial_state is None:
        num_units = params['num_units']
    dropout = params.get('dropout', None)
    cell_args = params.get('cell_args', {})
    if dp_masks is not None or dropout is None:
        dp_return_masks = None
    else:
        dp_return_masks = []
        distribution = tf.contrib.distributions.Uniform()
    cells = []

    with tf.variable_scope(scope, reuse=reuse):
        for i in range(num_cells):
            if dropout is not None:
                assert (type(dropout) is float and 0 < dropout and dropout <= 1.0)
                if dp_masks is not None:
                    dp = dp_masks[i]
                else:
                    if num_dp > 1:
                        sample = distribution.sample((tf.shape(inputs)[0] / num_dp, num_units))
                        sample = tf.concat([sample] * num_dp, axis=0)
                    else:
                        sample = distribution.sample((tf.shape(inputs)[0], num_units))
                    # Shape is not well defined without reshaping
                    sample = tf.reshape(sample, (-1, num_units))
                    mask = tf.cast(sample < dropout, dtype) / dropout
                    dp = mask
                    dp_return_masks.append(mask)
            else:
                dp = None

            if i == 0:
                num_inputs = inputs.get_shape()[-1]
            else:
                num_inputs = num_units
            cell = cell_type(
                num_units,
                dropout_mask=dp,
                dtype=dtype,
                num_inputs=num_inputs,
                weights_scope='{0}_{1}'.format(params['cell_type'], i),
                **cell_args)

            cells.append(cell)

        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, state = tf.nn.dynamic_rnn(
            multi_cell,
            tf.cast(inputs, dtype),
            initial_state=initial_state,
            dtype=dtype,
            time_major=False)

    return outputs, dp_return_masks

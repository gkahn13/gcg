import tensorflow as tf


def fully_connected_weight_norm(inputs, num_outputs, global_step_tensor=None, activation_fn=None,
                                weights_initializer=tf.random_normal_initializer(0, 0.05), init_scale=1.,
                                trainable=True, scope=None, reuse=None):
    assert (global_step_tensor is not None)
    x = inputs
    with tf.variable_scope(scope, default_name='wn_fc', reuse=reuse):
        V = tf.get_variable('V', [int(x.get_shape()[1]), num_outputs], tf.float32, weights_initializer,
                            trainable=trainable)
        g = tf.get_variable('g', [num_outputs], tf.float32,
                            tf.random_normal_initializer(0, 0.05))  # TODO: make nan init
        b = tf.get_variable('b', [num_outputs], tf.float32, tf.random_normal_initializer(0, 0.05))

        def init_weight_norm():
            # data based initialization of parameters
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(x, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g_new = scale_init
            b_new = -m_init * scale_init
            x_init = tf.reshape(scale_init, [1, num_outputs]) * (x_init - tf.reshape(m_init, [1, num_outputs]))
            return x_init, g_new, b_new

        def weight_norm(x):
            tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = tf.reshape(scaler, [1, num_outputs]) * x + tf.reshape(b, [1, num_outputs])

            return x, g, b

        x, g_new, b_new = tf.cond(tf.equal(global_step_tensor, 0), init_weight_norm, lambda: weight_norm(x))

        g_ass = tf.assign(g, g_new)
        b_ass = tf.assign(b, b_new)
        for ass in (g_ass, b_ass):
            tf.add_to_collection(tf.GraphKeys.INIT_OP, ass)

        if activation_fn is not None:
            x = activation_fn(x)

        return x


def conv2d_weight_norm(inputs, num_outputs, kernel_size, stride=1, padding='SAME', data_format=None,
                       global_step_tensor=None, activation_fn=None,
                       weights_initializer=tf.random_normal_initializer(0, 0.05), init_scale=1., trainable=True,
                       scope=None, reuse=None):
    assert (global_step_tensor is not None)
    x = inputs

    if type(kernel_size) == int:
        kernel_size = [kernel_size, kernel_size]
    if type(stride) == int:
        stride = [stride, stride]

    with tf.variable_scope(scope, default_name='wn_conv2d', reuse=reuse):
        V = tf.get_variable('V', kernel_size + [int(x.get_shape()[-1]), num_outputs], tf.float32, weights_initializer,
                            trainable=trainable)
        g = tf.get_variable('g', [num_outputs], tf.float32,
                            tf.random_normal_initializer(0, 0.05))  # TODO: make nan init
        b = tf.get_variable('b', [num_outputs], tf.float32, tf.random_normal_initializer(0, 0.05))

        def init_weight_norm():
            # data based initialization of parameters
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(x, V_norm, [1] + stride + [1], padding, data_format=data_format)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g_new = scale_init
            b_new = -m_init * scale_init
            x_init = tf.reshape(scale_init, [1, 1, 1, num_outputs]) * (
            x_init - tf.reshape(m_init, [1, 1, 1, num_outputs]))
            return x_init, g_new, b_new

        def weight_norm(x):
            tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_outputs]) * tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], padding, data_format=data_format), b)

            return x, g, b

        x, g_new, b_new = tf.cond(tf.equal(global_step_tensor, 0), init_weight_norm, lambda: weight_norm(x))

        g_ass = tf.assign(g, g_new)
        b_ass = tf.assign(b, b_new)
        for ass in (g_ass, b_ass):
            tf.add_to_collection(tf.GraphKeys.INIT_OP, ass)

        if activation_fn is not None:
            x = activation_fn(x)

        return x


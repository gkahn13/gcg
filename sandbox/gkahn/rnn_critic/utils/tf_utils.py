import tensorflow as tf

##################
### Optimizing ###
##################

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)

##################
### Operations ###
##################

def repeat_2d(x, reps, axis):
    assert(axis == 0 or axis == 1)

    if axis == 1:
        x = tf.transpose(x)

    static_shape = list(x.get_shape())
    dyn_shape = tf.shape(x)
    x_repeat = tf.reshape(tf.tile(x, [1, reps]), (dyn_shape[0] * reps, dyn_shape[1]))
    if static_shape[0].value is not None:
        static_shape[0] = tf.Dimension(static_shape[0].value *reps)
    x_repeat.set_shape(static_shape)

    if axis == 1:
        x_repeat = tf.transpose(x_repeat)

    return x_repeat

def batch_outer_product(X, Y):
    """
    :param X: [N, U]
    :param Y: [N, V]
    """
    # tf.assert_equal(tf.shape(X)[0], tf.shape(Y)[0])

    X_batch = tf.expand_dims(X, 2) # [N, U, 1]
    Y_batch = tf.expand_dims(Y, 1) # [N, 1, V]
    results = tf.batch_matmul(X_batch, Y_batch) # [N, U, V]

    return results

def batch_outer_product_2d(X, Y):
    """
    :param X: [N, U]
    :param Y: [N, V]
    :return [N, U * V]
    """
    U = X.get_shape()[1].value
    V = Y.get_shape()[1].value
    assert(U is not None)
    assert(V is not None)

    return tf.mul(tf.tile(X, (1, V)), repeat_2d(Y, U, 1))

def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
        matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
        matrices with the same batch dimension).
        dtype: Data type to use. The Tensors in `matrices` must match this dtype.
    Returns:
        A matrix with the input matrices stacked along its main diagonal, having
        shape [..., \sum_i N_i, \sum_i M_i].
  """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(0,
                [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
                 [(row_before_length, row_after_length)]])))
    blocked = tf.concat(-2, row_blocks)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked

if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    tf.set_random_seed(0)

    ### repeat_2d test
    a = tf.constant(np.random.random((2, 4)))
    a0 = repeat_2d(a, 2, 0)
    a1 = repeat_2d(a, 2, 1)

    sess = tf.Session()
    a_eval, a0_eval, a1_eval = sess.run([a, a0, a1])
    print('\nrepeat 2d test')
    print('a:\n{0}'.format(a_eval))
    print('a0\n{0}'.format(a0_eval))
    print('a1\n{0}'.format(a1_eval))

    ### test batch outer
    a = tf.constant(np.random.random((3, 2)))
    b = tf.constant(np.random.randint(0, 2, (3, 2)).astype(np.float64))
    ab_outer = tf.reshape(batch_outer_product(b, a), (a.get_shape()[0].value, -1))
    ab_outer_2d = batch_outer_product_2d(a, b)

    a_eval, b_eval, ab_outer_eval, ab_outer_2d_eval = sess.run([a, b, ab_outer, ab_outer_2d])
    print('\nbatch outer test')
    print('a:\n{0}'.format(a_eval))
    print('b:\n{0}'.format(b_eval))
    print('ab_outer:\n{0}'.format(ab_outer_eval))
    print('ab_outer_2d:\n{0}'.format(ab_outer_2d_eval))

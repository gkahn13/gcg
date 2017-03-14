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

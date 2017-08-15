import tensorflow as tf
assert(tf.__version__ == '1.2.1' or tf.__version__ == '0.12.1' or tf.__version__ == '0.11.0')

def global_variables_initializer():
    if tf.__version__ == '1.2.1' or tf.__version__ == '0.12.1':
        return tf.global_variables_initializer()
    elif tf.__version__ == '0.11.0':
        return tf.initialize_all_variables()
    else:
        raise Exception

def global_variables_collection_name():
    if tf.__version__ == '1.2.1' or tf.__version__ == '0.12.1':
        return tf.GraphKeys.GLOBAL_VARIABLES
    elif tf.__version__ == '0.11.0':
        return tf.GraphKeys.VARIABLES
    else:
        raise Exception

def trainable_variables_collection_name():
    return tf.GraphKeys.TRAINABLE_VARIABLES

def variables_initializer(vars):
    if tf.__version__ == '1.2.1' or tf.__version__ == '0.12.1':
        return tf.variables_initializer(vars)
    elif tf.__version__ == '0.11.0':
        return tf.initialize_variables(vars)
    else:
        raise Exception

def split(value, num_splits, axis=0):
    if tf.__version__ == '1.2.1':
        return tf.split(value, num_splits, axis=axis)
    else:
        return tf.split(axis, num_splits, value)

def concat(values, axis):
    if tf.__version__ == '1.2.1':
        return tf.concat(values, axis)
    else:
        return tf.concat(axis, values)

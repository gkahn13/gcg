import tensorflow as tf
assert(tf.__version__ == '1.2.1')

def global_variables_initializer():
    return tf.global_variables_initializer()

def global_variables_collection_name():
    return tf.GraphKeys.GLOBAL_VARIABLES

def trainable_variables_collection_name():
    return tf.GraphKeys.TRAINABLE_VARIABLES

def variables_initializer(vars):
    return tf.variables_initializer(vars)

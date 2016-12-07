"""Definition of Xception model introduced by F. Chollet.

Usage:
  with slim.arg_scope(xception.xception_arg_scope()):
    outputs, end_points = xception.xception(inputs)
@@xception
"""

import tensorflow as tf
slim = tf.contrib.slim


# =========================================================================== #
# Xception implementation (clean)
# =========================================================================== #
def xception(inputs,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             reuse=None,
             scope='xception'):
    """Xception model from https://arxiv.org/pdf/1610.02357v2.pdf

    The default image size used to train this network is 299x299.
    """

    # end_points collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    with tf.variable_scope(scope, 'xception', [inputs]):
        # Block 1.
        end_point = 'block1'
        with tf.variable_scope(end_point):
            net = slim.conv2d(inputs, 32, [3, 3], padding='VALID', scope='conv1')
            net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv2')
        end_points[end_point] = net

        # Residual block 2.
        end_point = 'block2'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 128, [1, 1], stride=2, activation_fn=None, scope='res')
            net = slim.separable_convolution2d(net, 128, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 128, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Residual block 3.
        end_point = 'block3'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 256, [1, 1], stride=2, activation_fn=None, scope='res')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 256, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 256, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Residual block 4.
        end_point = 'block4'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 728, [1, 1], stride=2, activation_fn=None, scope='res')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Middle flow blocks.
        for i in range(8):
            end_point = 'block' + str(i + 5)
            with tf.variable_scope(end_point):
                res = net
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv1')
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv2')
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv3')
                net = res + net
            end_points[end_point] = net

        # Exit flow: blocks 13 and 14.
        end_point = 'block13'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 1024, [1, 1], stride=2, activation_fn=None, scope='res')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv1')
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 1024, [3, 3], 1, activation_fn=None, scope='sepconv2')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        end_point = 'block14'
        with tf.variable_scope(end_point):
            net = slim.separable_convolution2d(net, 1536, [3, 3], 1, scope='sepconv1')
            net = slim.separable_convolution2d(net, 2048, [3, 3], 1, scope='sepconv2')
        end_points[end_point] = net

        # Global averaging.
        end_point = 'dense'
        with tf.variable_scope(end_point):
            net = tf.reduce_mean(net, [1, 2], name='reduce_avg')
            logits = slim.fully_connected(net, 1000, activation_fn=None)

            end_points['logits'] = logits
            end_points['predictions'] = prediction_fn(logits, scope='Predictions')

        return logits, end_points
xception.default_image_size = 299


def xception_arg_scope(weight_decay=0.00001, stddev=0.1):
    """Defines the default Xception arg scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.

    Returns:
      An `arg_scope` to use for the xception model.
    """
    batch_norm_params = {
      # Decay for the moving averages.
      'decay': 0.9997,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_convolution2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d, slim.separable_convolution2d],
            padding='SAME',
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                               mode='FAN_IN',
                                                                               uniform=False),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params) as sc:
                return sc


# =========================================================================== #
# Xception implementation (Keras hack!)
# =========================================================================== #
def xception_keras_arg_scope(hdf5_file, weight_decay=0.00001):
    """Defines an Xception arg scope which loads layers weights from a Keras
    HDF5 file.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the xception model.
    """
    # Default batch norm parameters.
    batch_norm_params = {
      'decay': 0.9997,
      'epsilon': 0.001,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    # Batch norm parameters from HDF5 file.
    def bn_params():
        bn_params.idx += 1
    bn_params.idx = 0

    # Conv2d weights from HDF5 file.
    def conv2d_weights():
        conv2d_weights.idx += 1
    conv2d_weights.idx = 0

    # Separable conv2d weights from HDF5 file.
    def sepconv2d_weights():
        sepconv2d_weights.idx += 1
    sepconv2d_weights.idx = 0

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_convolution2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d, slim.separable_convolution2d],
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params) as sc:
                return sc


def xception_keras(inputs,
                   hdf5_file=None,
                   num_classes=1000,
                   is_training=True,
                   dropout_keep_prob=0.5,
                   prediction_fn=slim.softmax,
                   reuse=None,
                   scope='xception_keras'):
    """Xception model from https://arxiv.org/pdf/1610.02357v2.pdf

    The default image size used to train this network is 299x299.
    """
    # Default batch norm. parameters.
    batch_norm_params = {
        'center': True,
        'scale': False,
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    end_points = {}

    # Read weights from HDF5 file.
    def bn_params():
        bn_params.idx += 1
        params = batch_norm_params.copy()

        # Batch norm. parameters.
        k = 'batchnormalization_%i' % bn_params.idx
        kb = 'batchnormalization_%i_gamma' % bn_params.idx
        km = 'batchnormalization_%i_running_mean' % bn_params.idx
        kv = 'batchnormalization_%i_running_std' % bn_params.idx
        param['param_initializers'] = {
            'beta': hdf5_file[k][kb][:],
            'moving_mean': hdf5_file[k][km][:],
            'moving_variance': hdf5_file[k][kv][:],
        }
        return params
    bn_params.idx = 0

    def conv2d_weights():
        init = tf.contrib.layers.variance_scaling_initializer()

        def _initializer(shape, dtype, partition_info=None):
            conv2d_weights.idx += 1
            print('Conv2d:', shape)
            k = 'convolution2d_%i' % conv2d_weights.idx
            kw = 'convolution2d_%i_W' % conv2d_weights.idx
            weights = hdf5_file[k][kw][:]
            print('Conv2d keras:', weights.shape)
            return weights.astype(dtype)
            # return init(shape, dtype, partition_info)
    conv2d_weights.idx = 0

    def sepconv2d_weights():
        init = tf.contrib.layers.variance_scaling_initializer()

        def _initializer(shape, dtype, partition_info=None):
            sepconv2d_weights.idx += 1
            print('SepConv2d:', shape)
            k = 'separableconvolution2d_%i' % sepconv2d_weights.idx
            kd = 'separableconvolution2d_%i_depthwise_kernel' % sepconv2d_weights.idx
            kp = 'separableconvolution2d_%i_pointwise_kernel' % sepconv2d_weights.idx
            weights = hdf5_file[k][kd][:]
            weights = hdf5_file[k][kp][:]
            print('SepConv2d keras:', weights.shape)
            return weights.astype(dtype)
            # return init(shape, dtype, partition_info)
    sepconv2d_weights.idx = 0
    sepconv2d_weights.subidx = 0

    def dense_weights():
        init = tf.contrib.layers.variance_scaling_initializer()

        def _initializer(shape, dtype, partition_info=None):
            dense_weights.idx += 1
            print('Dense:', shape)
            k = 'dense_%i' % dense_weights.idx
            kw = 'dense_%i_W' % dense_weights.idx
            weights = hdf5_file[k][kw][:]
            return weights.astype(dtype)
            # return init(shape, dtype, partition_info)
    dense_weights.idx = 1

    def dense_biases():
        init = tf.contrib.layers.variance_scaling_initializer()

        def _initializer(shape, dtype, partition_info=None):
            dense_biases.idx += 1
            print('Dense:', shape)
            k = 'dense_%i' % dense_biases.idx
            kb = 'dense_%i_b' % dense_biases.idx
            biases = hdf5_file[k][kb][:]
            return biases.astype(dtype)
            # return init(shape, dtype, partition_info)
    dense_biases.idx = 1

    with tf.variable_scope(scope, 'xception', [inputs]):
        # Block 1.
        end_point = 'block1'
        with tf.variable_scope(end_point):
            net = slim.conv2d(inputs, 32, [3, 3], padding='VALID', scope='conv1',
                              weights_initializer=conv2d_weights(),
                              normalizer_params=bn_params())
            net = slim.conv2d(net, 64, [3, 3], padding='VALID', scope='conv2',
                              weights_initializer=conv2d_weights(),
                              normalizer_params=bn_params())
        end_points[end_point] = net

        # Residual block 2.
        end_point = 'block2'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 128, [1, 1], stride=2, activation_fn=None, scope='res',
                              weights_initializer=conv2d_weights(),
                              normalizer_params=bn_params())
            net = slim.separable_convolution2d(net, 128, [3, 3], 1, scope='sepconv1',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.separable_convolution2d(net, 128, [3, 3], 1, activation_fn=None, scope='sepconv2',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Residual block 3.
        end_point = 'block3'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 256, [1, 1], stride=2, activation_fn=None, scope='res',
                              weights_initializer=conv2d_weights(),
                              normalizer_params=bn_params())
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 256, [3, 3], 1, scope='sepconv1',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.separable_convolution2d(net, 256, [3, 3], 1, activation_fn=None, scope='sepconv2',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Residual block 4.
        end_point = 'block4'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 728, [1, 1], stride=2, activation_fn=None, scope='res',
                              weights_initializer=conv2d_weights(),
                              normalizer_params=bn_params())
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, scope='sepconv1',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv2',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        # Middle flow blocks.
        for i in range(8):
            end_point = 'block' + str(i + 5)
            with tf.variable_scope(end_point):
                res = net
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv1',
                                                   weights_initializer=sepconv2d_weights(),
                                                   normalizer_params=bn_params())
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv2',
                                                   weights_initializer=sepconv2d_weights(),
                                                   normalizer_params=bn_params())
                net = tf.nn.relu(net)
                net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None,
                                                   scope='sepconv3',
                                                   weights_initializer=sepconv2d_weights(),
                                                   normalizer_params=bn_params())
                net = res + net
            end_points[end_point] = net

        # Exit flow: blocks 13 and 14.
        end_point = 'block13'
        with tf.variable_scope(end_point):
            res = slim.conv2d(net, 1024, [1, 1], stride=2, activation_fn=None, scope='res',
                              weights_initializer=conv2d_weights(),
                              normalizer_params=bn_params())
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 728, [3, 3], 1, activation_fn=None, scope='sepconv1',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = tf.nn.relu(net)
            net = slim.separable_convolution2d(net, 1024, [3, 3], 1, activation_fn=None, scope='sepconv2',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool')
            net = res + net
        end_points[end_point] = net

        end_point = 'block14'
        with tf.variable_scope(end_point):
            net = slim.separable_convolution2d(net, 1536, [3, 3], 1, scope='sepconv1',
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
            net = slim.separable_convolution2d(net, 2048, [3, 3], 1, scope='sepconv2',,
                                               weights_initializer=sepconv2d_weights(),
                                               normalizer_params=bn_params())
        end_points[end_point] = net

        # Global averaging.
        end_point = 'dense'
        with tf.variable_scope(end_point):
            net = tf.reduce_mean(net, [1, 2], name='reduce_avg')
            logits = slim.fully_connected(net, 1000, activation_fn=None,
                                          weights_initializer=dense_weights(),
                                          biases_initializer=dense_biases())
            end_points['logits'] = logits
            end_points['predictions'] = prediction_fn(logits, scope='Predictions')

        return logits, end_points
xception_keras.default_image_size = 299

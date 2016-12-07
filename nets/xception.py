"""Definition of Xception model introduced by F. Chollet.

Usage:
  with slim.arg_scope(xception.xception_arg_scope()):
    outputs, end_points = xception.xception(inputs)
@@xception
"""

import tensorflow as tf
slim = tf.contrib.slim


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
    """Defines the default InceptionV3 arg scope.

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


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a variant of the CIFAR-10 model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)


def atrousnet_same(images, num_classes=43, is_training=False,
                   dropout_keep_prob=0.5,
                   prediction_fn=slim.softmax,
                   scope='CifarNet'):
    """Creates a model using Dilated-Atrous convolutions.

    Args:
        images: A batch of `Tensors` of size [batch_size, height, width, channels].
        num_classes: the number of classes in the dataset.
        is_training: specifies whether or not we're currently training the model.
            This variable will determine the behaviour of the dropout layer.
        dropout_keep_prob: the percentage of activation values that are retained.
        prediction_fn: a function to get predictions out of logits.
        scope: Optional variable_scope.

    Returns:
        logits: the pre-softmax activations, a tensor of size
            [batch_size, `num_classes`]
        end_points: a dictionary from components of the network to the corresponding
            activation.
        """
    end_points = {}

    with tf.variable_scope(scope, 'AtrousNet', [images, num_classes]):

        net = slim.conv2d(images, 64, [3, 3], padding='SAME',
                          weights_regularizer=None,
                          scope='conv1')
        end_points['conv1'] = net
        net = slim.conv2d(net, 128, [3, 3], rate=2, padding='SAME',
                          weights_regularizer=None,
                          scope='conv2')
        end_points['conv2'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool2', padding='SAME')

        net = slim.conv2d(net, 192, [3, 3], rate=3, padding='SAME',
                          weights_regularizer=None,
                          scope='conv3')
        end_points['conv3'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool3', padding='SAME')

        net = slim.conv2d(net, 256, [3, 3], rate=4, padding='SAME',
                          # weights_regularizer=None,
                          scope='conv4')
        end_points['conv4'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool4', padding='SAME')

        net = slim.conv2d(net, 512, [1, 1],
                          # weights_regularizer=None,
                          # normalizer_fn=None,
                          scope='conv5')
        end_points['conv5'] = net
        net = slim.dropout(net, dropout_keep_prob,
                           is_training=is_training,
                           scope='dropout1')
        net = slim.conv2d(net, num_classes+1, [1, 1],
                          biases_initializer=tf.zeros_initializer,
                          weights_initializer=trunc_normal(1 / 512.0),
                          # weights_regularizer=None,
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='conv6')
        end_points['conv6'] = net

        # Background filter...
        net_back = slim.conv2d(net, 1, [1, 1],
                               biases_initializer=tf.zeros_initializer,
                               weights_initializer=trunc_normal(1 / 512.0),
                               # weights_regularizer=None,
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='conv_back')
        net_back = tf.nn.elu(net_back) + 1
        end_points['BackNet'] = net_back

        # Apply filteting to logits output.
        # Brings more weights on logits with proper background.
        net_back = tf.concat(3, [net_back] * (num_classes+1))
        net = tf.mul(net, net_back)
        # Pixel dropout.
        net_shape = net.get_shape()
        noise_shape = tf.pack([net_shape[0], net_shape[1], net_shape[2], 1])
        net = slim.dropout(net, 0.4,
                           noise_shape=noise_shape,
                           is_training=is_training,
                           scope='dropout_pixels')

        # Pixel predictions on the image.
        end_points['PredictionsFull'] = tf.nn.softmax(net)
        # Global average pooling of logits.
        logits = tf.reduce_mean(net, [1, 2], name='pool7')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points
atrousnet_same.default_image_size = 32


def atrousnet_valid(images, num_classes=43, is_training=False,
                    dropout_keep_prob=0.5,
                    prediction_fn=slim.softmax,
                    scope='CifarNet'):
    """Creates a model using Dilated-Atrous convolutions.

    Args:
        images: A batch of `Tensors` of size [batch_size, height, width, channels].
        num_classes: the number of classes in the dataset.
        is_training: specifies whether or not we're currently training the model.
            This variable will determine the behaviour of the dropout layer.
        dropout_keep_prob: the percentage of activation values that are retained.
        prediction_fn: a function to get predictions out of logits.
        scope: Optional variable_scope.

    Returns:
        logits: the pre-softmax activations, a tensor of size
            [batch_size, `num_classes`]
        end_points: a dictionary from components of the network to the corresponding
            activation.
        """
    end_points = {}

    with tf.variable_scope(scope, 'AtrousNet', [images, num_classes]):

        net = slim.conv2d(images, 64, [3, 3], padding='VALID',
                          weights_regularizer=None,
                          scope='conv1')
        end_points['conv1'] = net
        net = slim.conv2d(net, 128, [3, 3], rate=2, padding='VALID',
                          weights_regularizer=None,
                          scope='conv2')
        end_points['conv2'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool2', padding='SAME')

        net = slim.conv2d(net, 192, [3, 3], rate=3, padding='VALID',
                          weights_regularizer=None,
                          scope='conv3')
        end_points['conv3'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool3', padding='SAME')

        net = slim.conv2d(net, 256, [3, 3], rate=4, padding='VALID',
                          # weights_regularizer=None,
                          scope='conv4')
        end_points['conv4'] = net
        # net = slim.max_pool2d(net, [3, 3], 1, scope='pool4', padding='SAME')

        net = slim.conv2d(net, 512, [1, 1],
                          # weights_regularizer=None,
                          # normalizer_fn=None,
                          scope='conv5')
        end_points['conv5'] = net
        # Features dropout.
        net = slim.dropout(net, dropout_keep_prob,
                           is_training=is_training,
                           scope='dropout1')
        net = slim.conv2d(net, num_classes+1, [1, 1],
                          biases_initializer=tf.zeros_initializer,
                          weights_initializer=trunc_normal(1 / 512.0),
                          # weights_regularizer=None,
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='conv6')
        end_points['conv6'] = net

        # Background filtering...
        net_back = slim.conv2d(net, 1, [1, 1],
                               biases_initializer=tf.zeros_initializer,
                               weights_initializer=trunc_normal(1 / 512.0),
                               # weights_regularizer=None,
                               activation_fn=None,
                               normalizer_fn=None,
                               scope='conv_back')
        net_back = tf.nn.elu(net_back) + 1
        end_points['BackNet'] = net_back

        # Apply filteting to logits output.
        # Brings more weights on logits with proper background.
        net_back = tf.concat(3, [net_back] * (num_classes+1))
        net = tf.mul(net, net_back)

        # Pixel dropout.
        noise_shape = net.get_shape()
        noise_shape[3] = 1
        net = slim.dropout(net, 0.4,
                           noise_shape=noise_shape,
                           is_training=is_training,
                           scope='dropout_pixels')

        # Pixel predictions on the image.
        end_points['PredictionsFull'] = tf.nn.softmax(net)
        # Global average pooling of logits.
        logits = tf.reduce_mean(net, [1, 2], name='pool7')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points
atrousnet_valid.default_image_size = 32


def atrousnet_same_arg_scope(weight_decay=0.004):
    """Defines the default argument scope.

    Args:
        weight_decay: The weight decay to use for regularizing the model.
    Returns:
        An `arg_scope` to use for the inception v3 model.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.43),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            # weights_regularizer=None,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            activation_fn=tf.nn.relu) as sc:
        return sc


def atrousnet_valid_arg_scope(weight_decay=0.004):
    """Defines the default argument scope.

    Args:
        weight_decay: The weight decay to use for regularizing the model.
    Returns:
        An `arg_scope` to use for the inception v3 model.
    """
    batch_norm_params = {
        # Decay for the moving averages.
        'scale': False,
        'decay': 0.9997,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }
    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.uniform_unit_scaling_initializer(factor=1.43),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            # weights_regularizer=None,
            # weights_initializer=tf.truncated_normal_initializer(stddev=5e-2),
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params,
            activation_fn=tf.nn.relu) as sc:
        return sc

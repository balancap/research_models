# Copyright 2017 Paul Balanca. All Rights Reserved.
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
"""Contains model definitions for MobileNets
CAFFE-like implementation! Should reproduce the performance announced,
i.e. around 0.7032 Top-1 Classification.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models import model
from models_slim import custom_layers

slim = tf.contrib.slim

# VGG mean parameters.
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_SCALING = 0.017


# =========================================================================== #
# MobileNets class.
# =========================================================================== #
class MobileNetsModel(model.Model):
    def __init__(self, model='mobilenets', width_multiplier=1.0):
        super(MobileNetsModel, self).__init__(model, 224, 64, 0.005)
        self.width_multiplier = width_multiplier

    def inference(self, images, num_classes,
                  is_training=True, data_format='NCHW', data_type=tf.float32):
        # Define VGG using functional slim definition
        arg_scope = mobilenets_arg_scope(is_training=is_training, data_format=data_format)
        with slim.arg_scope(arg_scope):
            return mobilenets(images, num_classes, self.width_multiplier,
                              is_training=is_training)

    def pre_rescaling(self, images, is_training=True):
        return mobilenets_pre_rescaling(images, is_training)


# =========================================================================== #
# Functional definition.
# =========================================================================== #
def mobilenets_pre_rescaling(images, is_training=True):
    """Rescales an images Tensor before feeding the network
    Input tensor supposed to be in [0, 256) range.
    """
    # Rescale to [-1,1] instead of [0, 1)
    # images *= 1. / 255.
    # images = tf.subtract(images, 0.5)
    # images = tf.multiply(images, 2.0)
    mean = tf.constant([_R_MEAN, _G_MEAN, _B_MEAN], dtype=images.dtype)
    images = images - mean
    images = images * _SCALING
    return images


def mobilenets_arg_scope(weight_decay=0.0000004,
                         data_format='NCHW',
                         batch_norm_decay=0.9997,
                         batch_norm_epsilon=0.00001,
                         is_training=True):
    """Defines the default arg scope for MobileNets models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': True,
        'data_format': data_format,
        'is_training': is_training,
    }
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d,
                         slim.fully_connected,
                         custom_layers.depthwise_convolution2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d,
                 custom_layers.depthwise_convolution2d],
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=tf.nn.relu,
                normalizer_fn=normalizer_fn,
                normalizer_params=normalizer_params,
                padding='SAME'):
            # Data format scope...
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d,
                                 slim.max_pool2d, slim.avg_pool2d,
                                 custom_layers.depthwise_convolution2d,
                                 custom_layers.pad2d,
                                 custom_layers.channel_to_last,
                                 custom_layers.spatial_squeeze,
                                 custom_layers.spatial_mean],
                                data_format=data_format) as sc:
                return sc


def mobilenets(inputs,
               num_classes=1000,
               width_multiplier=1.0,
               is_training=True,
               dropout_keep_prob=0.5,
               scope='MobileNets'):
    """MobileNets implementation.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions and end_points dict.
    """
    kernel_size = [3, 3]
    padding = [(kernel_size[0]-1)//2, (kernel_size[1]-1)//2]

    def mobilenet_block(net, num_out_channels, stride=[1, 1],
                        scope=None):
        """Basic MobileNet block combining:
         - depthwise conv + BN + relu
         - 1x1 conv + BN + relu
        """
        with tf.variable_scope(scope, 'block', [net]) as sc:
            num_out_channels = int(num_out_channels * width_multiplier)
            if stride[0] == 1 and stride[1] == 1:
                # Depthwise convolution with stride=1
                net = custom_layers.depthwise_convolution2d(
                    net, kernel_size,
                    depth_multiplier=1, stride=stride,
                    scope='conv_dw')
            else:
                # Mimic CAFFE padding if stride > 1.
                net = custom_layers.pad2d(net, pad=padding)
                net = custom_layers.depthwise_convolution2d(
                    net, kernel_size, padding='VALID',
                    depth_multiplier=1, stride=stride,
                    scope='conv_dw')
            # Pointwise convolution.
            net = slim.conv2d(net, num_out_channels, [1, 1],
                              scope='conv_pw')
            return net

    with tf.variable_scope(scope, 'MobileNets', [inputs]) as sc:
        end_points = {}
        # First full convolution...
        net = custom_layers.pad2d(inputs, pad=padding)
        net = slim.conv2d(net, 32, kernel_size, stride=[2, 2],
                          padding='VALID', scope='conv1')
        # net = slim.conv2d(inputs, 32, [ksize, ksize], stride=[2, 2], scope='conv1')
        # Then, MobileNet blocks!
        net = mobilenet_block(net, 64, scope='block2')
        net = mobilenet_block(net, 128, stride=[2, 2], scope='block3')
        net = mobilenet_block(net, 128, scope='block4')
        net = mobilenet_block(net, 256, stride=[2, 2], scope='block5')
        net = mobilenet_block(net, 256, scope='block6')
        net = mobilenet_block(net, 512, stride=[2, 2], scope='block7')
        # Intermediate blocks...
        for i in range(5):
            net = mobilenet_block(net, 512, scope='block%i' % (i+8))
        # Final blocks.
        net = mobilenet_block(net, 1024, stride=[2, 2], scope='block13')
        net = mobilenet_block(net, 1024, scope='block14')
        # Spatial pooling + fully connected layer.
        net = custom_layers.spatial_mean(net, keep_dims=True, scope='spatial_mean14')
        net = slim.conv2d(net, 1000, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          normalizer_params=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='conv_fc15')
        net = custom_layers.spatial_squeeze(net)
        # net = slim.fully_connected(net, 1000,  scope='fc15')

        # Logits padding...
        net = custom_layers.pad_logits(net, pad=(num_classes - 1000, 0))
        return net, end_points
mobilenets.default_image_size = 224

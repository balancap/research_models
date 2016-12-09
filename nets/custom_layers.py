"""Some custom layers, implementing random and ludicrous ideas
I sometimes have.
"""

import math
import numpy as np
import tensorflow as tf

import functools

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

slim = tf.contrib.slim


# =========================================================================== #
# Separable convolution 2d with difference padding.
# =========================================================================== #
@add_arg_scope
def separable_convolution2d_diffpad(
        inputs,
        num_outputs,
        kernel_size,
        depth_multiplier,
        stride=1,
        padding='SAME',
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer,
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    """Adds a depth-separable 2D convolution with optional batch_norm layer.
    This op first performs a depthwise convolution that acts separately on
    channels, creating a variable called `depthwise_weights`. If `num_outputs`
    is not None, it adds a pointwise convolution that mixes channels, creating a
    variable called `pointwise_weights`. Then, if `batch_norm_params` is None,
    it adds bias to the result, creating a variable called 'biases', otherwise
    it adds a batch normalization layer. It finally applies an activation function
    to produce the end result.

    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_outputs: the number of pointwise convolution output filters. If is
            None, then we skip the pointwise convolution stage.
        kernel_size: a list of length 2: [kernel_height, kernel_width] of
            of the filters. Can be an int if both values are the same.
        depth_multiplier: the number of depthwise convolution output channels for
            each input channel. The total number of depthwise convolution output
            channels will be equal to `num_filters_in * depth_multiplier`.
        stride: a list of length 2: [stride_height, stride_width], specifying the
            depthwise convolution stride. Can be an int if both strides are the same.
        padding: one of 'VALID' or 'SAME'.
        activation_fn: activation function, set to None to skip it and maintain
            a linear activation.
        normalizer_fn: normalization function to use instead of `biases`. If
            `normalizer_fn` is provided then `biases_initializer` and
            `biases_regularizer` are ignored and `biases` are not created nor added.
            default set to None for no normalizer function
        normalizer_params: normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        variables_collections: optional list of collections for all the variables or
            a dictionay containing a different list of collection per variable.
        outputs_collections: collection to add the outputs.
        trainable: whether or not the variables should be trainable or not.
        scope: Optional scope for variable_scope.
    Returns:
        A `Tensor` representing the output of the operation.
    """
    with variable_scope.variable_scope(
            scope, 'SeparableConv2dPad', [inputs], reuse=reuse) as sc:
        # TOFIX: Hard set padding and multiplier...
        padding = 'SAME'
        depth_multiplier = 1

        dtype = inputs.dtype.base_dtype
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
        weights_collections = utils.get_variable_collections(
                variables_collections, 'weights')

        depthwise_shape = [kernel_h, kernel_w,
                           num_filters_in, depth_multiplier]
        depthwise_weights = variables.model_variable(
                'depthwise_weights',
                shape=depthwise_shape,
                dtype=dtype,
                initializer=weights_initializer,
                regularizer=weights_regularizer,
                trainable=trainable,
                collections=weights_collections)
        strides = [1, stride_h, stride_w, 1]

        # Classic depthwise_conv2d with SAME padding (i.e. zero padding).
        outputs = nn.depthwise_conv2d(inputs, depthwise_weights, strides, padding)

        # Fix padding according too the difference rule. Dirty shit!
        # diff_padding = variables.local_variable(tf.zeros_like(outputs))
        diff_padding = variables.variable(
            'diff_padding',
            shape=outputs.get_shape(),
            dtype=dtype,
            initializer=tf.constant_initializer(),
            regularizer=None,
            trainable=False,
            collections=weights_collections)

        # Bottom and top fixing...
        # print(diff_padding[:, 0, :, :].get_shape())
        # print(depthwise_weights[0, 0, :, 0].get_shape())
        op1 = diff_padding[:, 0, :, :].assign(
            outputs[:, 0, :, :] * (depthwise_weights[0, 0, :, 0] +
                                   depthwise_weights[0, 1, :, 0] +
                                   depthwise_weights[0, 2, :, 0]))
        op2 = diff_padding[:, -1, :, :].assign(
            outputs[:, -1, :, :] * (depthwise_weights[-1, 0, :, 0] +
                                    depthwise_weights[-1, 1, :, 0] +
                                    depthwise_weights[-1, 2, :, 0]))
        # Bottom and top fixing...
        op3 = diff_padding[:, :, 0, :].assign(
            outputs[:, :, 0, :] * (depthwise_weights[0, 0, :, 0] +
                                   depthwise_weights[1, 0, :, 0] +
                                   depthwise_weights[2, 0, :, 0]))
        op4 = diff_padding[:, :, -1, :].assign(
            outputs[:, :, -1, :] * (depthwise_weights[0, -1, :, 0] +
                                    depthwise_weights[1, -1, :, 0] +
                                    depthwise_weights[2, -1, :, 0]))
        diff_padding1 = control_flow_ops.with_dependencies([op1, op2, op3, op4],
                                                           diff_padding)

        # Fix double addition in corners.
        op5 = diff_padding[:, 0, 0, :].assign(
            diff_padding1[:, 0, 0, :] - outputs[:, 0, 0, :] * depthwise_weights[0, 0, :, 0])
        op6 = diff_padding[:, -1, 0, :].assign(
            diff_padding1[:, -1, 0, :] - outputs[:, -1, 0, :] * depthwise_weights[-1, 0, :, 0])
        op7 = diff_padding[:, 0, -1, :].assign(
            diff_padding1[:, 0, -1, :] - outputs[:, 0, -1, :] * depthwise_weights[0, -1, :, 0])
        op8 = diff_padding[:, -1, -1, :].assign(
            diff_padding1[:, -1, -1, :] - outputs[:, -1, -1, :] * depthwise_weights[-1, -1, :, 0])
        diff_padding2 = control_flow_ops.with_dependencies([op5, op6, op7, op8],
                                                           diff_padding)

        # # Update padding!
        outputs = outputs + diff_padding2

        # Adding pointwise convolution.
        if num_outputs is not None:
            # Full separable convolution: Depthwise followed by pointwise convolution.
            pointwise_shape = [1, 1, depth_multiplier * num_filters_in, num_outputs]
            pointwise_weights = variables.model_variable(
                    'pointwise_weights',
                    shape=pointwise_shape,
                    dtype=dtype,
                    initializer=weights_initializer,
                    regularizer=weights_regularizer,
                    trainable=trainable,
                    collections=weights_collections)

            outputs = nn.conv2d(outputs, pointwise_weights,
                                strides=(1, 1, 1, 1), padding='SAME')

        else:
            # Depthwise convolution only.
            num_outputs = depth_multiplier * num_filters_in

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases_collections = utils.get_variable_collections(
                        variables_collections, 'biases')
                biases = variables.model_variable('biases',
                                                  shape=[num_outputs,],
                                                  dtype=dtype,
                                                  initializer=biases_initializer,
                                                  regularizer=biases_regularizer,
                                                  collections=biases_collections)
                outputs = nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


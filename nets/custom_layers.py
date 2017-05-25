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
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.ops import variables as tf_variables

slim = tf.contrib.slim


# =========================================================================== #
# Extension of TensorFlow common layers.
# =========================================================================== #
@add_arg_scope
def l2_normalization(
        inputs,
        scaling=False,
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        data_format='NHWC',
        trainable=True,
        scope=None):
    """Implement L2 normalization on every feature (i.e. spatial normalization).

    Should be extended in some near future to other dimensions, providing a more
    flexible normalization framework.

    Args:
      inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
      scaling: whether or not to add a post scaling operation along the dimensions
        which have been normalized.
      scale_initializer: An initializer for the weights.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: collection to add the outputs.
      data_format:  NHWC or NCHW data format.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    """

    with variable_scope.variable_scope(
            scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            # norm_dim = tf.range(1, inputs_rank-1)
            norm_dim = tf.range(inputs_rank-1, inputs_rank)
            params_shape = inputs_shape[-1:]
        elif data_format == 'NCHW':
            # norm_dim = tf.range(2, inputs_rank)
            norm_dim = tf.range(1, 2)
            params_shape = (inputs_shape[1])

        # Normalize along spatial dimensions.
        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling.
        if scaling:
            scale_collections = utils.get_variable_collections(
                variables_collections, 'scale')
            scale = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=scale_initializer,
                                             collections=scale_collections,
                                             trainable=trainable)
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs, scale)
                # outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          trainable=True,
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net


@add_arg_scope
def channel_to_last(inputs, data_format='NHWC', scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a single output format whatever the input data format.

    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net


@add_arg_scope
def concat_channels(l_inputs, data_format='NHWC', scope=None):
    """Concat a list of tensors on the channel axis.

    Args:
      inputs: List Tensors;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'concat_channels', l_inputs):
        if data_format == 'NHWC':
            net = tf.concat(l_inputs, axis=3)
        elif data_format == 'NCHW':
            net = tf.concat(l_inputs, axis=1)
        return net


@add_arg_scope
def spatial_mean(inputs, keep_dims=False, data_format='NHWC', scope=None):
    """Average tensor along spatial dimensions.

    Args:
      inputs: Input tensor;
      keep_dims: Keep spatial dimensions?
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'spatial_mean', [inputs]):
        axes = [1, 2] if data_format == 'NHWC' else [2, 3]
        net = tf.reduce_mean(inputs, axes, keep_dims=keep_dims)
        return net


@add_arg_scope
def spatial_squeeze(inputs, data_format='NHWC', scope=None):
    """Squeeze spatial dimensions, if possible.

    Args:
      inputs: Input tensor;
      data_format: NHWC or NCHW.
    """
    with tf.name_scope(scope, 'spatial_squeeze', [inputs]):
        axes = [1, 2] if data_format == 'NHWC' else [2, 3]
        net = tf.squeeze(inputs, axes)
        return net


@add_arg_scope
def ksize_for_squeezing(inputs, default_ksize=[1024, 1024], data_format='NHWC'):
    """Get the correct kernel size for squeezing input tensor.
    """
    shape = inputs.get_shape().as_list()
    kshape = shape[1:3] if data_format == 'NHWC' else shape[2:]
    if kshape[0] is None or kshape[1] is None:
        kernel_size_out = default_ksize
    else:
        kernel_size_out = [min(kshape[0], default_ksize[0]),
                           min(kshape[1], default_ksize[1])]
    return kernel_size_out


# =========================================================================== #
# Depthwise convolution 2d with data format option.
# =========================================================================== #
@add_arg_scope
def depthwise_convolution2d(
        inputs,
        kernel_size,
        depth_multiplier=1,
        stride=1,
        padding='SAME',
        rate=1,
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        data_format='NHWC',
        scope=None):
    """Adds a depthwise 2D convolution with optional batch_norm layer.
    This op performs a depthwise convolution that acts separately on
    channels, creating a variable called `depthwise_weights`. Then,
    if `normalizer_fn` is None,
    it adds bias to the result, creating a variable called 'biases', otherwise,
    the `normalizer_fn` is applied. It finally applies an activation function
    to produce the end result.
    Args:
        inputs: A tensor of size [batch_size, height, width, channels].
        num_outputs: The number of pointwise convolution output filters. If is
          None, then we skip the pointwise convolution stage.
        kernel_size: A list of length 2: [kernel_height, kernel_width] of
          of the filters. Can be an int if both values are the same.
        depth_multiplier: The number of depthwise convolution output channels for
          each input channel. The total number of depthwise convolution output
          channels will be equal to `num_filters_in * depth_multiplier`.
        stride: A list of length 2: [stride_height, stride_width], specifying the
          depthwise convolution stride. Can be an int if both strides are the same.
        padding: One of 'VALID' or 'SAME'.
        rate: A list of length 2: [rate_height, rate_width], specifying the dilation
          rates for atrous convolution. Can be an int if both rates are the same.
          If any value is larger than one, then both stride values need to be one.
        activation_fn: Activation function. The default value is a ReLU function.
          Explicitly set it to None to skip it and maintain a linear activation.
        normalizer_fn: Normalization function to use instead of `biases`. If
          `normalizer_fn` is provided then `biases_initializer` and
          `biases_regularizer` are ignored and `biases` are not created nor added.
          default set to None for no normalizer function
        normalizer_params: Normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: Whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        variables_collections: Optional list of collections for all the variables or
          a dictionary containing a different list of collection per variable.
        outputs_collections: Collection to add the outputs.
        trainable: Whether or not the variables should be trainable or not.
        scope: Optional scope for variable_scope.
    Returns:
        A `Tensor` representing the output of the operation.
    """
    with variable_scope.variable_scope(scope, 'DepthwiseConv2d', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        # Actually apply depthwise conv instead of separable conv.
        dtype = inputs.dtype.base_dtype
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        if data_format == 'NHWC':
            num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
            strides = [1, stride_h, stride_w, 1]
        else:
            num_filters_in = inputs.get_shape().as_list()[1]
            strides = [1, 1, stride_h, stride_w]

        weights_collections = utils.get_variable_collections(
            variables_collections, 'weights')

        # Depthwise weights variable.
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

        outputs = nn.depthwise_conv2d(inputs, depthwise_weights,
                                      strides, padding,
                                      rate=utils.two_element_tuple(rate),
                                      data_format=data_format)
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
                                                  trainable=trainable,
                                                  collections=biases_collections)
                outputs = nn.bias_add(outputs, biases)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
depthwise_conv2d = depthwise_convolution2d


@add_arg_scope
def pad_logits(logits, pad=(0, 0)):
    """Pad logits Tensor, to deal with different
    number of classes.
    """
    shape = logits.get_shape().as_list()
    dtype = logits.dtype
    l = [logits]
    if pad[0] > 0:
        a = tf.constant(dtype.min, dtype, (shape[0], pad[0]))
        l = [a] + l
    if pad[1] > 0:
        a = tf.constant(dtype.min, dtype, (shape[0], pad[1]))
        l = l + [a]
    output = tf.concat(l, axis=1)
    return output


# =========================================================================== #
# Separable convolution 2d with data format option.
# =========================================================================== #
def _model_variable_getter(getter, name, shape=None, dtype=None,
                           initializer=None, regularizer=None, trainable=True,
                           collections=None, caching_device=None,
                           partitioner=None, rename=None, use_resource=None,
                           **_):
    """Getter that uses model_variable for compatibility with core layers."""
    short_name = name.split('/')[-1]
    if rename and short_name in rename:
        name_components = name.split('/')
        name_components[-1] = rename[short_name]
        name = '/'.join(name_components)
    return variables.model_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, collections=collections, trainable=trainable,
        caching_device=caching_device, partitioner=partitioner,
        custom_getter=getter, use_resource=use_resource)


def _build_variable_getter(rename=None):
    """Build a model variable getter that respects scope getter and renames."""
    # VariableScope will nest the getters
    def layer_variable_getter(getter, *args, **kwargs):
        kwargs['rename'] = rename
        return _model_variable_getter(getter, *args, **kwargs)
    return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
    """Adds variable (or all its parts) to all collections with that name."""
    collections = utils.get_variable_collections(
            collections_set, collections_name) or []
    variables_list = [variable]
    if isinstance(variable, tf_variables.PartitionedVariable):
        variables_list = [v for v in variable]
    for collection in collections:
        for var in variables_list:
            if var not in ops.get_collection(collection):
                ops.add_to_collection(collection, var)


@add_arg_scope
def separable_convolution2d(
        inputs,
        num_outputs,
        kernel_size,
        depth_multiplier,
        stride=1,
        padding='SAME',
        rate=1,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        data_format='NHWC',
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
        inputs: A tensor of size [batch_size, height, width, channels].
        num_outputs: The number of pointwise convolution output filters. If is
            None, then we skip the pointwise convolution stage.
        kernel_size: A list of length 2: [kernel_height, kernel_width] of
            of the filters. Can be an int if both values are the same.
        depth_multiplier: The number of depthwise convolution output channels for
            each input channel. The total number of depthwise convolution output
            channels will be equal to `num_filters_in * depth_multiplier`.
        stride: A list of length 2: [stride_height, stride_width], specifying the
            depthwise convolution stride. Can be an int if both strides are the same.
        padding: One of 'VALID' or 'SAME'.
        rate: A list of length 2: [rate_height, rate_width], specifying the dilation
            rates for a'trous convolution. Can be an int if both rates are the same.
            If any value is larger than one, then both stride values need to be one.
        activation_fn: Activation function. The default value is a ReLU function.
            Explicitly set it to None to skip it and maintain a linear activation.
        normalizer_fn: Normalization function to use instead of `biases`. If
            `normalizer_fn` is provided then `biases_initializer` and
            `biases_regularizer` are ignored and `biases` are not created nor added.
            default set to None for no normalizer function
        normalizer_params: Normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: Whether or not the layer and its variables should be reused. To be
            able to reuse the layer scope must be given.
        variables_collections: Optional list of collections for all the variables or
            a dictionary containing a different list of collection per variable.
        outputs_collections: Collection to add the outputs.
        trainable: Whether or not the variables should be trainable or not.
        scope: Optional scope for variable_scope.
    Returns:
        A `Tensor` representing the output of the operation.
    """
    layer_variable_getter = _build_variable_getter(
            {'bias': 'biases',
             'depthwise_kernel': 'depthwise_weights',
             'pointwise_kernel': 'pointwise_weights'})

    with variable_scope.variable_scope(
            scope, 'SeparableConv2d', [inputs], reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)

        if num_outputs is not None:
            channel_format = 'channels_last' if data_format == 'NHWC' else 'channels_first'
            # Apply separable conv using the SeparableConvolution2D layer.
            layer = convolutional_layers.SeparableConvolution2D(
                    filters=num_outputs,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    data_format=channel_format,
                    dilation_rate=utils.two_element_tuple(rate),
                    activation=None,
                    depth_multiplier=depth_multiplier,
                    use_bias=not normalizer_fn and biases_initializer,
                    depthwise_initializer=weights_initializer,
                    pointwise_initializer=weights_initializer,
                    bias_initializer=biases_initializer,
                    depthwise_regularizer=weights_regularizer,
                    pointwise_regularizer=weights_regularizer,
                    bias_regularizer=biases_regularizer,
                    activity_regularizer=None,
                    trainable=trainable,
                    name=sc.name,
                    dtype=inputs.dtype.base_dtype,
                    _scope=sc,
                    _reuse=reuse)
            outputs = layer.apply(inputs)

            # Add variables to collections.
            _add_variable_to_collections(layer.depthwise_kernel,
                                         variables_collections, 'weights')
            _add_variable_to_collections(layer.pointwise_kernel,
                                         variables_collections, 'weights')
            if layer.bias:
                _add_variable_to_collections(layer.bias,
                                             variables_collections, 'biases')

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            outputs = depthwise_convolution2d(
                inputs,
                kernel_size,
                depth_multiplier,
                stride,
                padding,
                rate,
                activation_fn,
                normalizer_fn,
                normalizer_params,
                weights_initializer,
                weights_regularizer,
                biases_initializer,
                biases_regularizer,
                reuse,
                variables_collections,
                outputs_collections,
                trainable,
                data_format,
                scope=None)
            return outputs

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
separable_conv2d = separable_convolution2d


# =========================================================================== #
# Convolution leaders layers...
# =========================================================================== #
@add_arg_scope
def depthwise_leaders_convolution2d(
        inputs,
        kernel_size,
        stride=1,
        padding='SAME',
        rates=[1, 2],
        pooling_sizes=[3, 1],
        pooling_type='MAX',
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        data_format='NHWC',
        scope=None):
    """Adds a depthwise 2D convolution with optional batch_norm layer.
    This op performs a depthwise convolution that acts separately on
    channels, creating a variable called `depthwise_weights`. Then,
    if `normalizer_fn` is None,
    it adds bias to the result, creating a variable called 'biases', otherwise,
    the `normalizer_fn` is applied. It finally applies an activation function
    to produce the end result.
    Args:
        inputs: A tensor of size [batch_size, height, width, channels].
        num_outputs: The number of pointwise convolution output filters. If is
          None, then we skip the pointwise convolution stage.
        kernel_size: A list of length 2: [kernel_height, kernel_width] of
          of the filters. Can be an int if both values are the same.
        depth_multiplier: The number of depthwise convolution output channels for
          each input channel. The total number of depthwise convolution output
          channels will be equal to `num_filters_in * depth_multiplier`.
        stride: A list of length 2: [stride_height, stride_width], specifying the
          depthwise convolution stride. Can be an int if both strides are the same.
        padding: One of 'VALID' or 'SAME'.
        rate: A list of length 2: [rate_height, rate_width], specifying the dilation
          rates for atrous convolution. Can be an int if both rates are the same.
          If any value is larger than one, then both stride values need to be one.
        activation_fn: Activation function. The default value is a ReLU function.
          Explicitly set it to None to skip it and maintain a linear activation.
        normalizer_fn: Normalization function to use instead of `biases`. If
          `normalizer_fn` is provided then `biases_initializer` and
          `biases_regularizer` are ignored and `biases` are not created nor added.
          default set to None for no normalizer function
        normalizer_params: Normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: Whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        variables_collections: Optional list of collections for all the variables or
          a dictionary containing a different list of collection per variable.
        outputs_collections: Collection to add the outputs.
        trainable: Whether or not the variables should be trainable or not.
        scope: Optional scope for variable_scope.
    Returns:
        A `Tensor` representing the output of the operation.
    """
    with variable_scope.variable_scope(scope, 'DepthwiseLeadersConv2d', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        if data_format == 'NHWC':
            num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
            strides = [1, stride_h, stride_w, 1]
        else:
            num_filters_in = inputs.get_shape().as_list()[1]
            strides = [1, 1, stride_h, stride_w]
        # Depthwise weights + biases variables.
        depth_multiplier = 1
        num_outputs = depth_multiplier * num_filters_in

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
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          trainable=trainable,
                                          collections=biases_collections)

        # Perform the convolution at different rates.
        outputs = []
        for i, rate in enumerate(rates):
            # Depthwise conv.
            net = nn.depthwise_conv2d(inputs, depthwise_weights,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      rate=utils.two_element_tuple(rate),
                                      data_format=data_format)
            # Add bias + abs. val.
            net = tf.abs(nn.bias_add(net, biases, data_format=data_format))
            # Pooling...
            if pooling_sizes[i] > 1:
                net = tf.nn.pool(net,
                                 [pooling_sizes[i], pooling_sizes[i]],
                                 pooling_type,
                                 padding='SAME',
                                 data_format=data_format)
            outputs.append(net)
        # Fuse different rates/scales.
        net = None
        for i, o in enumerate(outputs):
            if net is None:
                # First in the list...
                net = o
            else:
                # MAX or AVG pooling...
                if pooling_type == 'MAX':
                    net = tf.maximum(net, o)
                else:
                    net += o * pooling_sizes[i]**2
        # Pooling => for stride > 1
        if stride_h > 1 or stride_w > 1:
            # net = tf.nn.pool(net,
            #                  [1, 1],
            #                  pooling_type,
            #                  padding='SAME',
            #                  strides=[stride_h, stride_w],
            #                  data_format=data_format)
            net = tf.nn.max_pool(net,
                                 [1, 1, 1, 1],
                                 padding='SAME',
                                 strides=strides,
                                 data_format=data_format)
        # (Batch normalization)...
        normalizer_params = normalizer_params or {}
        net = normalizer_fn(net, **normalizer_params)

        # Split into two parts: positive and negative extreme.
        net_p = slim.bias_add(net, data_format=data_format, scope='bias_positive')
        net_p = activation_fn(net_p)
        # net_p = slim.dropout(net_p, 0.5, is_training=True, scope='dropout_p')

        net_m = slim.bias_add(-net, data_format=data_format, scope='bias_negative')
        net_m = activation_fn(net_m)
        # net_m = slim.dropout(net_m, 0.5, is_training=True, scope='dropout_m')

        # Concat the final result...
        outputs = concat_channels([net_p, net_m], data_format=data_format)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
depthwise_leaders_conv2d = depthwise_leaders_convolution2d


@add_arg_scope
def leaders_convolution2d(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        padding='SAME',
        rates=[1, 2],
        pooling_sizes=[3, 1],
        pooling_type='MAX',
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        data_format='NHWC',
        scope=None):
    """Adds a depthwise 2D convolution with optional batch_norm layer.
    This op performs a depthwise convolution that acts separately on
    channels, creating a variable called `depthwise_weights`. Then,
    if `normalizer_fn` is None,
    it adds bias to the result, creating a variable called 'biases', otherwise,
    the `normalizer_fn` is applied. It finally applies an activation function
    to produce the end result.
    Args:
        inputs: A tensor of size [batch_size, height, width, channels].
        num_outputs: The number of pointwise convolution output filters. If is
          None, then we skip the pointwise convolution stage.
        kernel_size: A list of length 2: [kernel_height, kernel_width] of
          of the filters. Can be an int if both values are the same.
        depth_multiplier: The number of depthwise convolution output channels for
          each input channel. The total number of depthwise convolution output
          channels will be equal to `num_filters_in * depth_multiplier`.
        stride: A list of length 2: [stride_height, stride_width], specifying the
          depthwise convolution stride. Can be an int if both strides are the same.
        padding: One of 'VALID' or 'SAME'.
        rate: A list of length 2: [rate_height, rate_width], specifying the dilation
          rates for atrous convolution. Can be an int if both rates are the same.
          If any value is larger than one, then both stride values need to be one.
        activation_fn: Activation function. The default value is a ReLU function.
          Explicitly set it to None to skip it and maintain a linear activation.
        normalizer_fn: Normalization function to use instead of `biases`. If
          `normalizer_fn` is provided then `biases_initializer` and
          `biases_regularizer` are ignored and `biases` are not created nor added.
          default set to None for no normalizer function
        normalizer_params: Normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: Whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        variables_collections: Optional list of collections for all the variables or
          a dictionary containing a different list of collection per variable.
        outputs_collections: Collection to add the outputs.
        trainable: Whether or not the variables should be trainable or not.
        scope: Optional scope for variable_scope.
    Returns:
        A `Tensor` representing the output of the operation.
    """
    with variable_scope.variable_scope(scope, 'LeadersConv2d', [inputs],
                                       reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        if data_format == 'NHWC':
            num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)
            strides = [1, stride_h, stride_w, 1]
        else:
            num_filters_in = inputs.get_shape().as_list()[1]
            strides = [1, 1, stride_h, stride_w]

        # Conv weights + biases variables.
        weights_collections = utils.get_variable_collections(
            variables_collections, 'weights')
        depthwise_shape = [kernel_h, kernel_w,
                           num_filters_in, num_outputs]
        depthwise_weights = variables.model_variable(
            'leaders_weights',
            shape=depthwise_shape,
            dtype=dtype,
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable,
            collections=weights_collections)
        biases_collections = utils.get_variable_collections(
            variables_collections, 'biases')
        biases = variables.model_variable('biases',
                                          shape=[num_outputs,],
                                          dtype=dtype,
                                          initializer=biases_initializer,
                                          regularizer=biases_regularizer,
                                          trainable=trainable,
                                          collections=biases_collections)

        # Perform the convolution at different rates.
        outputs = []
        for i, rate in enumerate(rates):
            # Depthwise conv.
            net = nn.convolution(inputs, depthwise_weights,
                                 strides=[1, 1],
                                 padding='SAME',
                                 dilation_rate=utils.two_element_tuple(rate),
                                 data_format=data_format)
            # Add bias + abs. val.
            net = tf.abs(nn.bias_add(net, biases, data_format=data_format))
            # Pooling...
            if pooling_sizes[i] > 1:
                net = tf.nn.pool(net,
                                 [pooling_sizes[i], pooling_sizes[i]],
                                 pooling_type,
                                 padding='SAME',
                                 data_format=data_format)
            outputs.append(net)
        # Fuse different rates/scales.
        net = None
        for i, o in enumerate(outputs):
            if net is None:
                # First in the list...
                net = o
            else:
                # MAX or AVG pooling...
                if pooling_type == 'MAX':
                    net = tf.maximum(net, o)
                else:
                    net += o * pooling_sizes[i]**2
        # Pooling => for stride > 1
        if stride_h > 1 or stride_w > 1:
            # net = tf.nn.pool(net,
            #                  [1, 1],
            #                  pooling_type,
            #                  padding='SAME',
            #                  strides=[stride_h, stride_w],
            #                  data_format=data_format)
            net = tf.nn.max_pool(net,
                                 [1, 1, 1, 1],
                                 padding='SAME',
                                 strides=strides,
                                 data_format=data_format)
        # (Batch normalization)...
        normalizer_params = normalizer_params or {}
        net = normalizer_fn(net, **normalizer_params)

        # Split into two parts: positive and negative extreme.
        net_p = slim.bias_add(net, data_format=data_format, scope='bias_positive')
        net_p = activation_fn(net_p)
        net_p = slim.dropout(net_p, 0.5, is_training=True, scope='dropout_p')

        net_m = slim.bias_add(-net, data_format=data_format, scope='bias_negative')
        net_m = activation_fn(net_m)
        net_m = slim.dropout(net_m, 0.5, is_training=True, scope='dropout_m')
        # Concat the final result...
        outputs = concat_channels([net_p, net_m], data_format=data_format)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
leaders_conv2d = leaders_convolution2d


# =========================================================================== #
# Block pointwise convolution.
# =========================================================================== #

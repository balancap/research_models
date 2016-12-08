"""Convert a Keras checkpoint an HDF5 file to TensorFlow file format.

Assume that the network built is a equivalent (or a sub-) to the Keras
definition.
"""

import h5py
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

# =========================================================================== #
# Main flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'model_name', 'xception', 'The name of the architecture to convert.')
tf.app.flags.DEFINE_string(
    'num_classes', 1000, 'Number of classes in the dataset.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to the Keras checkpoint to convert.')

FLAGS = tf.app.flags.FLAGS


# =========================================================================== #
# Main converting routine.
# =========================================================================== #
def main(_):
    # Load Keras weights.
    hdf5_keras = h5py.File(FLAGS.checkpoint_path, mode='r')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # Select the network.
        kwargs = {
            'hdf5_file': hdf5_keras,
            'weight_decay': 0.0
        }
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            is_training=False,
            num_classes=FLAGS.num_classes, **kwargs)

        # Image placeholder and model.
        shape = (1, network_fn.default_image_size, network_fn.default_image_size, 3)
        img_input = tf.placeholder(shape=shape, dtype=tf.float32)

        # Create model.
        logits, end_points = network_fn(img_input)

        init_op = tf.initialize_all_variables()
        with tf.Session() as session:
            # Run the init operation.
            session.run(init_op)

            # Restore model checkpoint.
            saver = tf.train.Saver()
            ckpt_path = FLAGS.checkpoint_path.replace('.h5', '.ckpt')
            # saver.restore(session, ckpt_file)
            saver.save(session, ckpt_path, write_meta_graph=False)


if __name__ == '__main__':
    tf.app.run()


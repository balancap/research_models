## A few commands...

DATASET_DIR=../traffic-signs-data/GTSRB_size32
python tf_convert_data.py \
    --dataset_name=gtsrb_32_transform \
    --dataset_dir="${DATASET_DIR}"

rm events* graph* model* checkpoint
mv events* graph* model* checkpoint ./idsianet_log6


# ===========================================================================
# TinyNet
# ===========================================================================
DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
CHECKPOINT_PATH=logs/
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32_transform \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --labels_offset=1 \
    --model_name=tinynet \
    --optimizer=rmsprop \
    --rmsprop_momentum=0.9 \
    --rmsprop_decay=0.9 \
    --opt_epsilon=1.0 \
    --learning_rate=1.0 \
    --num_epochs_per_decay=0.1 \
    --learning_rate_decay_factor=0.9 \
    --weight_decay=0.000005 \
    --batch_size=64

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=tinynet

# ===========================================================================
# Inception v3
# ===========================================================================
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
DATASET_DIR=../datasets/ImageNet
TRAIN_DIR=./logs/inception_v3
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
CHECKPOINT_PATH=./checkpoints/inception_v3.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=4


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/logs
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3


# ===========================================================================
# VGG 16 and 19
# ===========================================================================
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_19.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --labels_offset=1 \
    --dataset_split_name=validation \
    --model_name=vgg_19


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_16.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --labels_offset=1 \
    --dataset_split_name=validation \
    --model_name=vgg_16 \
    --max_num_batches=10


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_16.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/Training/log_vgg_1
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=vgg_16 \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.0001 \
    --batch_size=32


# ===========================================================================
# Xception
# ===========================================================================
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/Training/logs_xception
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.0001 \
    --batch_size=32

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception \
    --labels_offset=1 \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=1


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
CHECKPOINT_PATH=./logs/xception
DATASET_DIR=../datasets/ImageNet

CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=xception \
    --max_num_batches=10


CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.h5
python ckpt_keras_to_tensorflow.py \
    --model_name=xception_keras \
    --num_classes=1000 \
    --checkpoint_path=${CHECKPOINT_PATH}


# ===========================================================================
# Xception B-tree
# ===========================================================================
DATASET_DIR=/home/paul/Development/Datasets/ImageNet
TRAIN_DIR=./logs/log_xception_btree_1
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt

CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt

DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/Training/logs_xception_btree/log_018
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/logs_xception_btree/log_017/model.ckpt-6332
nohup python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception_btree \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --trainable_scopes=xception/block2/sepconv1/btree_conv_1x1,xception/block2/sepconv1/BatchNorm,xception/block2/sepconv2/btree_conv_1x1,xception/block2/sepconv2/BatchNorm,xception/block3/sepconv1/btree_conv_1x1,xception/block3/sepconv1/BatchNorm,xception/block3/sepconv2/btree_conv_1x1,xception/block3/sepconv2/BatchNorm,xception/block4/sepconv1/btree_conv_1x1,xception/block4/sepconv1/BatchNorm,xception/block4/sepconv2/btree_conv_1x1,xception/block4/sepconv2/BatchNorm,xception/block5/sepconv1/btree_conv_1x1,xception/block5/sepconv1/BatchNorm,xception/block5/sepconv2/btree_conv_1x1,xception/block5/sepconv2/BatchNorm,xception/block5/sepconv3/btree_conv_1x1,xception/block5/sepconv3/BatchNorm,xception/block6/sepconv1/btree_conv_1x1,xception/block6/sepconv1/BatchNorm,xception/block6/sepconv2/btree_conv_1x1,xception/block6/sepconv2/BatchNorm,xception/block6/sepconv3/btree_conv_1x1,xception/block6/sepconv3/BatchNorm,xception/block7/sepconv1/btree_conv_1x1,xception/block7/sepconv1/BatchNorm,xception/block7/sepconv2/btree_conv_1x1,xception/block7/sepconv2/BatchNorm,xception/block7/sepconv3/btree_conv_1x1,xception/block7/sepconv3/BatchNorm,xception/block8/sepconv1/btree_conv_1x1,xception/block8/sepconv1/BatchNorm,xception/block8/sepconv2/btree_conv_1x1,xception/block8/sepconv2/BatchNorm,xception/block8/sepconv3/btree_conv_1x1,xception/block8/sepconv3/BatchNorm,xception/block9/sepconv1/btree_conv_1x1,xception/block9/sepconv1/BatchNorm,xception/block9/sepconv2/btree_conv_1x1,xception/block9/sepconv2/BatchNorm,xception/block9/sepconv3/btree_conv_1x1,xception/block9/sepconv3/BatchNorm,xception/block10/sepconv1/btree_conv_1x1,xception/block10/sepconv1/BatchNorm,xception/block10/sepconv2/btree_conv_1x1,xception/block10/sepconv2/BatchNorm,xception/block10/sepconv3/btree_conv_1x1,xception/block10/sepconv3/BatchNorm,xception/block11/sepconv1/btree_conv_1x1,xception/block11/sepconv1/BatchNorm,xception/block11/sepconv2/btree_conv_1x1,xception/block11/sepconv2/BatchNorm,xception/block11/sepconv3/btree_conv_1x1,xception/block11/sepconv3/BatchNorm,xception/block12/sepconv1/btree_conv_1x1,xception/block12/sepconv1/BatchNorm,xception/block12/sepconv2/btree_conv_1x1,xception/block12/sepconv2/BatchNorm,xception/block12/sepconv3/btree_conv_1x1,xception/block5/sepconv3/BatchNorm \
    --ignore_missing_vars=True \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.000000001 \
    --optimizer=rmsprop \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.1 \
    --moving_average_decay=0.999 \
    --batch_size=16 &

DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=xception_btree \
    --moving_average_decay=0.9999 \
    --max_num_batches=10


# ===========================================================================
# Xception B-tree distillation
# ===========================================================================
DATASET_DIR=/home/paul/Development/Datasets/ImageNet
TRAIN_DIR=./logs/log_xception_btree_1
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt

DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/Training/logs_xception_btree_distill/log_005
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
nohup python distill_btree_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=xception \
    --model_btree=xception_btree \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --trainable_scopes=xception_btree/block2/sepconv1/btree_conv_1x1,xception_btree/block2/sepconv1/BatchNorm,xception_btree/block2/sepconv2/btree_conv_1x1,xception_btree/block2/sepconv2/BatchNorm,xception_btree/block3/sepconv1/btree_conv_1x1,xception_btree/block3/sepconv1/BatchNorm,xception_btree/block3/sepconv2/btree_conv_1x1,xception_btree/block3/sepconv2/BatchNorm,xception_btree/block4/sepconv1/btree_conv_1x1,xception_btree/block4/sepconv1/BatchNorm,xception_btree/block4/sepconv2/btree_conv_1x1,xception_btree/block4/sepconv2/BatchNorm \
    --distill_points=block2_1,block2_2,block3_1,block3_2,block4_1,block4_2 \
    --ignore_missing_vars=True \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.000000001 \
    --optimizer=rmsprop \
    --learning_rate=0.1 \
    --learning_rate_decay_factor=0.1 \
    --moving_average_decay=0.999 \
    --batch_size=16 &

DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=xception_btree \
    --moving_average_decay=0.9999 \
    --max_num_batches=10

# ===========================================================================
# Dception
# ===========================================================================
DATASET_DIR=../datasets/ImageNet
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/Training/logs_dception/2
CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=dception \
    --labels_offset=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.00001 \
    --learning_rate_decay_factor=0.94 \
    --optimizer=rmsprop \
    --learning_rate=0.005 \
    --batch_size=26

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=dception \
    --labels_offset=1 \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --weight_decay=0.00001 \
    --optimizer=rmsprop \
    --learning_rate=0.00005 \
    --batch_size=1

CHECKPOINT_PATH=./checkpoints/xception_weights_tf_dim_ordering_tf_kernels.ckpt
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --labels_offset=1 \
    --batch_size=16 \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=dception \
    --max_num_batches=10

# ===========================================================================
# MobileNets
# ===========================================================================
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/training/logs/mobilenet_pool_001
CHECKPOINT_PATH=./checkpoints/mobilenets.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --model_name=mobilenets_pool \
    --labels_offset=0 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=120 \
    --save_interval_secs=900 \
    --weight_decay=0.00001 \
    --learning_rate_decay_type=polynomial \
    --learning_rate_decay_factor=0.94 \
    --optimizer=rmsprop \
    --learning_rate=0.001 \
    --endend_learning_rate=0.00001 \
    --num_epochs_per_decay=20. \
    --moving_average_decay=0.9999 \
    --batch_size=192

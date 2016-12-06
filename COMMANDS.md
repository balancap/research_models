## A few commands...

DATASET_DIR=../traffic-signs-data/GTSRB_size32
python tf_convert_data.py \
    --dataset_name=gtsrb_32_transform \
    --dataset_dir="${DATASET_DIR}"

rm events* graph* model* checkpoint
mv events* graph* model* checkpoint ./idsianet_log6

# ===========================================================================
# CifarNet
# ===========================================================================
DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
CHECKPOINT_PATH=logs/log5/model.ckpt-897281
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32_transform \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --model_name=cifarnet \
    --optimizer=rmsprop \
    --learning_rate=0.04 \
    --num_epochs_per_decay=1. \
    --learning_rate_decay_factor=0.8 \
    --weight_decay=0.00001 \
    --batch_size=128

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=cifarnet


# ===========================================================================
# AtrousNet
# ===========================================================================
DATASET_DIR=../traffic-signs-data/GTSRB_size32
TRAIN_DIR=logs/
CHECKPOINT_PATH=logs/atrousnet_log2/model.ckpt-372595
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=gtsrb_32_transform \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --save_summaries_secs=60 \
    --save_interval_secs=60 \
    --labels_offset=1 \
    --model_name=atrousnet_same \
    --optimizer=adam \
    --rmsprop_momentum=0.9 \
    --rmsprop_decay=0.9 \
    --opt_epsilon=1.0 \
    --learning_rate=2.0 \
    --num_epochs_per_decay=0.5 \
    --learning_rate_decay_factor=0.9 \
    --weight_decay=0.00001 \
    --batch_size=256

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=atrousnet


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
# Xception
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
    --model_name=xception \
    --optimizer=rmsprop \
    --rmsprop_momentum=0.9 \
    --rmsprop_decay=0.9 \
    --opt_epsilon=1.0 \
    --learning_rate=1.0 \
    --num_epochs_per_decay=0.1 \
    --learning_rate_decay_factor=0.9 \
    --weight_decay=0.00005 \
    --batch_size=128

DATASET_DIR=../traffic-signs-data/GTSRB_size32
CHECKPOINT_FILE=logs
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=gtsrb_32 \
    --dataset_split_name=test \
    --model_name=xception

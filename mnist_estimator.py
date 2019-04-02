import json
import os
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
import sys

from tensorflow.python.keras.utils import *

import logging
import mnist_dataset

logging.getLogger().setLevel(logging.INFO)

FLAGS = app.flags.FLAGS
flags = app.flags

# =======================================================================
# Constant variables
# --work_dir=/data
# --data_dir=/data/data
# --output_dir=/data/output
#
# Note: Use this params as contant values
#       Do not set this params !!!
# =======================================================================
flags.DEFINE_string("work_dir", "/data", "Default work path")
flags.DEFINE_string("data_dir", "/data/data", "Default data path")
flags.DEFINE_string("output_dir", "/data/output", "Default output path")
flags.DEFINE_integer("num_gpus", 0, "Num of avaliable gpus")

# How many categories we are predicting from (0-9)
LABEL_DIMENSIONS = 10


def load_mnist_data():
    path = './mnist_data/mnist.npz'
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


def get_input():
    (train_images, train_labels), (test_images, test_labels) = load_mnist_data()
    TRAINING_SIZE = len(train_images)
    TEST_SIZE = len(test_images)

    train_images = np.asarray(train_images, dtype=np.float32) / 255

    # Convert the train images and add channels
    train_images = train_images.reshape((TRAINING_SIZE, 28, 28, 1))

    test_images = np.asarray(test_images, dtype=np.float32) / 255
    # Convert the train images and add channels
    test_images = test_images.reshape((TEST_SIZE, 28, 28, 1))

    train_labels = tf.keras.utils.to_categorical(train_labels, LABEL_DIMENSIONS)
    test_labels = tf.keras.utils.to_categorical(test_labels, LABEL_DIMENSIONS)

    # Cast the labels to floats, needed later
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    return train_images, train_labels, test_images, test_labels


def my_input_fn(data_dir='/data/data/mnist_tfrecords',
                subset='Train',
                num_shards=0,
                batch_size=4,
                use_distortion_for_training=False):
    """Create input graph for model.

    Args:
      data_dir: Directory where TFRecords representing the dataset are located.
      subset: one of 'train', 'validate' and 'eval'.
      num_shards: num of towers participating in data-parallel training.
      batch_size: total batch size for training to be divided by the number of
      shards.
      use_distortion_for_training: True to use distortions.
    Returns:
      three
    """
    with tf.device('/cpu:0'):
        # use_distortion = subset == 'train' and use_distortion_for_training
        use_distortion = False
        dataset = mnist_dataset.MnistDataSet(data_dir, subset, use_distortion)
        input_data, input_labels = dataset.make_batch(batch_size)
        labels = tf.one_hot(indices=input_labels, depth=10)  # config.cfg.TRAIN.CLASSES_NUMS)
        one_hot_labels = tf.cast(labels, tf.int32)

        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            num_shards = 1

        feature_shards = tf.split(input_data, num_shards)
        # label_shards = tf.sparse_split(sp_input=input_labels, num_split=num_shards, axis=0)
        label_shards = tf.split(one_hot_labels, num_shards)
        return feature_shards, label_shards


def build_model():
    inputs = tf.keras.Input(shape=(32, 32, 1))  # Returns a placeholder tensor
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(x)
    predictions = tf.keras.layers.Dense(LABEL_DIMENSIONS, activation=tf.nn.softmax)(x)

    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def input_fn(images, labels, repeat, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Shuffle, repeat, and batch the examples.
    SHUFFLE_SIZE = 100
    dataset = dataset.shuffle(SHUFFLE_SIZE).repeat(repeat).batch(batch_size)

    # Return the dataset.
    return dataset


def train():
    '''
    # pack tf_dist_conf
    if 'TF_CONFIG' in os.environ:
        tf_dist_conf = os.environ['TF_CONFIG']
        conf = json.loads(tf_dist_conf)
        if conf['task']['type'] == 'ps':
            is_ps = True
        else:
            is_ps = False

        if conf['task']['type'] == 'master':
            conf['task']['type'] = 'chief'

        conf['cluster']['chief'] = conf['cluster']['master']
        del conf['cluster']['master']          # delete all conf setting about 'master', trans to 'chief'
        print(conf)
        os.environ['TF_CONFIG'] = json.dumps(conf)
    else:
        print('tf_config not exists in os.environ, task over.')
        return
    '''

    model = build_model()

    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    '''
    if is_ps:
        distribution = tf.distribute.experimental.ParameterServerStrategy()
    else:
        distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    '''
    # config = tf.estimator.RunConfig(train_distribute=distribution)

    estimator = tf.keras.estimator.model_to_estimator(model, model_dir='/data/output/mnist_estimator_ckp')  # FLAGS.output_dir)

    # train_images, train_labels, test_images, test_labels = get_input()
    feature_shards, label_shards = my_input_fn()
    train_images = feature_shards  #['images']
    train_labels = label_shards #['labels']


    BATCH_SIZE = 16
    EPOCHS = 5
    STEPS = 2000

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: my_input_fn(batch_size=BATCH_SIZE),
                                        max_steps=STEPS)

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: my_input_fn(batch_size=BATCH_SIZE),
                                      steps=1,
                                      start_delay_secs=3)



    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec)


if __name__ == '__main__':
    train()

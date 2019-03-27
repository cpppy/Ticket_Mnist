import os
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

FLAGS = app.flags.FLAGS
flags = app.flags


# =======================================================================
flags.DEFINE_string("work_dir", "/data", "Default work path")
flags.DEFINE_string("data_dir", "/data/data", "Default data path")
flags.DEFINE_string("output_dir", "/data/output", "Default output path")
flags.DEFINE_integer("num_gpus", 0, "Num of avaliable gpus")

def load_mnist_data():
    path = '/data/data/mnist.npz'
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


def train():

  mnist = tf.keras.datasets.mnist

  (x_train, y_train),(x_test, y_test) = mnist.load_data('/data/data')
  x_train, x_test = x_train / 255.0, x_test / 255.0

  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  weights_path = '/data/data/mnist.h5'
  model.load_weights(weights_path)

  model.fit(x_train, y_train, epochs=5)
  output_path = os.path.join(FLAGS.output_dir, 'mnist.h5')
  model.save(output_path)

def main(_):
  train()

if __name__ == '__main__':
  app.run(main)
  #
  # (x_train, y_train), (x_test, y_test) = load_mnist_data()
  # print(y_test.shape)
  # print(y_train[0:3])



import os
import numpy as np
import tensorflow as tf
from absl import app
import data_generator
import parameter as params


def load_mnist_data():
    json_train_path = '/data/data/mnist_train_data/labels/train.json'
    json_val_path = '/data/data/mnist_train_data/labels/val.json'
    save_path = '/data/data/mnist_train_data/images'

    train_data = data_generator.DataGenerator(img_dirpath=save_path,
                                              json_path=json_train_path,
                                              img_w=params.img_w,
                                              img_h=params.img_h,
                                              batch_size=params.batch_size)
    train_data.build_data()
    train_sample_num = train_data.n

    val_data = data_generator.DataGenerator(img_dirpath=save_path,
                                            json_path=json_val_path,
                                            img_w=params.img_w,
                                            img_h=params.img_h,
                                            batch_size=params.batch_size)
    val_data.build_data()
    val_sample_num = val_data.n

    return train_data.next_batch(), \
           val_data.next_batch(), \
           train_sample_num, \
           val_sample_num


def train():

    train_data_gen, val_data_gen, train_sample_num, val_sample_num = load_mnist_data()

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint =tf.keras.callbacks.ModelCheckpoint(filepath='/data/output/mnist--{epoch:02d}--{val_acc:.3f}.h5',
                                                   monitor='val_acc',
                                                   save_best_only=False,
                                                   save_weights_only=True,
                                                   verbose=1,
                                                   mode='auto',
                                                   period=1)

    # weights_path = '/data/output/mnist.h5'
    # model.load_weights(weights_path)

    # model.fit(x_train, y_train, epochs=5)
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=train_sample_num // params.batch_size,
                        epochs=100,
                        callbacks=[checkpoint],
                        verbose=2,
                        # initial_epoch=0,
                        validation_data=val_data_gen,
                        validation_steps=val_sample_num // params.batch_size)


    output_path = os.path.join('/data/output', 'mnist.h5')
    model.save(output_path)


def main(_):
    train()


if __name__ == '__main__':
    app.run(main)

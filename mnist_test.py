import os
import cv2
import numpy as np
import tensorflow as tf


def rescale_to_cube(img):
    img_h, img_w = img.shape[0:2]
    cube_len = max(img_h, img_w)
    res_img = np.ones((cube_len, cube_len), dtype=np.uint8) * 255.0
    paste_pos = (int((cube_len-img_w)/2), int((cube_len-img_h)/2))
    res_img[paste_pos[1]:img_h+paste_pos[1], paste_pos[0]:img_w+paste_pos[00]] = img
    return res_img




if __name__=='__main__':

    weights_path = '/data/output/mnist--09--0.988.h5'

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

    model.load_weights(weights_path)


    img_dir = './test_data2'
    img_fn_list = os.listdir(img_dir)

    for img_fn in img_fn_list:
        img_path = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_path, 0)
        cv2.imshow('res', img_cv2)
        cv2.waitKey(0)
        img_cv2 = rescale_to_cube(img_cv2)
        img_cv2 = cv2.resize(img_cv2, (32, 32))

        img_pred = img_cv2.astype(np.float32)
        img_pred = img_pred/255.0

        img_pred = np.expand_dims(img_pred, axis=-1)
        input = np.expand_dims(img_pred, axis=0)

        print(input.shape)
        net_out_value = model.predict(input)
        print(net_out_value)
        res = np.argmax(net_out_value)
        print(res)



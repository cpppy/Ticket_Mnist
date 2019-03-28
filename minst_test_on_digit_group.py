import os
import cv2
import numpy as np
import tensorflow as tf
import pickle


def rescale_to_cube(img):
    img_h, img_w = img.shape[0:2]
    cube_len = max(img_h, img_w)
    res_img = np.ones((cube_len, cube_len), dtype=np.uint8) * 255.0
    paste_pos = (int((cube_len - img_w) / 2), int((cube_len - img_h) / 2))
    res_img[paste_pos[1]:img_h + paste_pos[1], paste_pos[0]:img_w + paste_pos[00]] = img
    return res_img


def draw_compare_result(img_cv2, pred_label):
    img_h, img_w = img_cv2.shape[0:2]
    canvas = np.ones(shape=(2 * img_h, img_w), dtype=np.uint8) * 200
    canvas[0:img_h, :] = img_cv2
    cv2.putText(img=canvas,
                text=pred_label,
                org=(5, img_h * 2 - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=0,
                thickness=2)
    display_img = canvas.copy()
    display_img = cv2.resize(display_img, (img_w * 10, img_h * 20))
    # cv2.imshow('compare_result', display_img)
    # cv2.waitKey(0)
    return display_img


if __name__ == '__main__':

    weights_path = '/data/output/mnist--16--0.988.h5'

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

    img_dir = './money_num_group'
    fn_list = os.listdir(img_dir)
    img_fn_list = []
    for fn in fn_list:
        if 'jpg' in fn:
            img_fn_list.append(fn)
    img_fn_list = sorted(img_fn_list)
    for img_fn in img_fn_list:
        print(img_fn)
        img_path = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_path)

        digit_list = []
        pickle_fn = img_fn.replace('jpg', 'pickle')
        with open(os.path.join(img_dir, pickle_fn), 'rb') as f:
            img_arr = pickle.load(f)
        for digit_img in img_arr:
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            digit_img = rescale_to_cube(digit_img)
            digit_img = cv2.resize(digit_img, (32, 32))

            img_pred = digit_img.astype(np.float32)
            img_pred = img_pred / 255.0

            img_pred = np.expand_dims(img_pred, axis=-1)
            input = np.expand_dims(img_pred, axis=0)

            # print(input.shape)
            net_out_value = model.predict(input)
            # print(net_out_value)
            res = np.argmax(net_out_value)
            digit_list.append(res)

        print('moeny: ', ''.join([str(elem) for elem in digit_list]))
        cv2.imshow('digit_group', img_cv2)
        cv2.waitKey(0)

        # compare_img = draw_compare_result(img_cv2, str(res))
        # cv2.imwrite(os.path.join('./compare_img', img_fn), compare_img)




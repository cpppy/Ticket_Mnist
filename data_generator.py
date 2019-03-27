import os
import numpy as np
import json
import numpy as np
import random
import parameter as params
import cv2


class DataGenerator:
    def __init__(self,
                 img_dirpath,
                 json_path,
                 img_w,
                 img_h,
                 batch_size):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.img_dirpath = img_dirpath  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)  # images list
        self.n = 0  # number of images
        self.indexes = []
        self.json_path = json_path
        self.cur_index = 0
        self.imgs = []
        self.texts = []

    ## samples
    def build_data(self):
        print("DataGenerator, build data ...")
        # load image_label_dict, {image_name:label}
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.img_text_dict = json.load(f)
        self.img_fn_list = [i for i, j in self.img_text_dict.items()]
        self.n = len(self.img_fn_list)
        print("sample size of current generator: ", self.n)
        self.indexes = list(range(self.n))
        random.shuffle(self.indexes)

    def next_sample(self):  ## index max -> 0
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        # load one image and its label
        img_idx = self.indexes[self.cur_index]
        img_fn = self.img_fn_list[img_idx]
        # print(os.path.join(self.img_dirpath, img_fn))
        img = cv2.imread(os.path.join(self.img_dirpath, img_fn), cv2.IMREAD_GRAYSCALE)
        # print(img.shape)

        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32)
        img = np.array(img / 255.0)
        value = int(self.img_text_dict[img_fn])
        return img, value

    def next_batch(self):  ## batch size
        while True:
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])  # (bs, 128, 64, 1)
            # TODO  Y_data ---> 0
            Y_data = np.zeros([self.batch_size], dtype=np.float)  # (bs, 9)

            for i in range(self.batch_size):
                img, value = self.next_sample()
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = value

            # dict
            inputs = {
                'the_input': X_data,  # (bs, 32, 32)
            }
            outputs = {'the_labels': Y_data}  # (bs, 10)
            # yield (inputs, outputs)
            yield X_data, Y_data





if __name__ == "__main__":
    json_train_path = '/data/data/mnist_train_data/labels/train.json'
    json_val_path = '/data/data/mnist_train_data/labels/val.json'
    save_path = '/data/data/mnist_train_data/images'

    train_data = DataGenerator(img_dirpath=save_path,
                               json_path=json_val_path,
                               img_w=params.img_w,
                               img_h=params.img_h,
                               batch_size=params.batch_size)
    train_data.build_data()

    inputs, outputs = train_data.next_batch().__next__()
    print(inputs)
    print(outputs)

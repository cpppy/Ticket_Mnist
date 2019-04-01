# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os
import tensorflow as tf
import re


class MnistDataSet(object):
    """Cifar10 data set.

    Described by http://www.cs.toronto.edu/~kriz/cifar.html.
    """

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset.lower() == 'train':
            re_patten = r'^train_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
        elif self.subset.lower() == 'validation':
            re_patten = r'^val_feature_\d{0,15}_\d{0,15}\.tfrecords\Z'
        else:
           raise ValueError('Invalid data subset "%s"' % self.subset)

        tfrecords_dir = '/data/data/mnist_tfrecords'
        tfrecords_list = [os.path.join(tfrecords_dir, tmp) for tmp in os.listdir(tfrecords_dir) if re.match(re_patten, tmp)]

        print('tfrecords_list: ', tfrecords_list)
        return tfrecords_list

        '''
        if self.subset in ['train']:
            #return [os.path.join(self.data_dir, 'train_feature_%d_%d.tfrecords' % (i*32, i*32 + 32))
            #    for i in range(3125)]
            files1 = [os.path.join(self.data_dir, 'train_feature_%d_%d.tfrecords' % (i*10000, i*10000 + 10000))
                for i in range(263)]
            files2 = [os.path.join(self.data_dir, 'train_feature_%d_%d.tfrecords' % (i*10000, i*10000 + 10000))
                for i in range(270, 420)]
            return files1+files2
        if self.subset in ['validation']:
            #return [os.path.join(self.data_dir, 'validation_feature_%d_%d.tfrecords' % (i*32, i*32 + 32))
            #    for i in range(313)]
            return [os.path.join(self.data_dir, 'train_feature_%d_%d.tfrecords' % (i*10000, i*10000 + 10000))
                for i in range(421, 431)]
        '''

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'images': tf.FixedLenFeature((), tf.string),
                'imagenames': tf.FixedLenFeature([1], tf.string),
                'labels': tf.FixedLenFeature((), tf.int64),
                # 'labels': tf.VarLenFeature(tf.int64),
            })
        image = tf.decode_raw(features['images'], tf.uint8)
        images = tf.reshape(image, [32, 32, 1])
        labels = features['labels']
        labels = tf.cast(labels, tf.int32)
   
        return images, labels

    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        #dataset = tf.contrib.data.TFRecordDataset(filenames).repeat()
        dataset = tf.data.TFRecordDataset(filenames).repeat()

        # Parse records.
        #dataset = dataset.map(
        #    self.parser, num_threads=8, output_buffer_size=2 * batch_size)
        dataset = dataset.map(self.parser,
                              num_parallel_calls=2)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = 32 #int(MnistDataSet.num_examples_per_epoch(self.subset) * 0.001)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 180
        elif subset == 'validation':
            return 20
        elif subset == 'eval':
            return 20
        else:
            raise ValueError('Invalid data subset "%s"' % subset)


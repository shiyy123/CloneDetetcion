import sys

import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/hope.csv"
TEST_URL = "http://download.tensorflow.org/data/test.csv"

CSV_COLUMN_NAMES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'same']
SAME = ['similar', 'dissimilar']


# 获取训练集和测试集的地址
def maybe_download():
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    test_path = tf.keras.utils.get_file(fname=TEST_URL.split('/')[-1],
                                        origin=TEST_URL)

    return train_path, test_path


# 获取训练集和测试集中的特征和标签
def load_data(y_name='same'):
    train_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1]]


def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('same')
    return features, label


def csv_input_fn(csv_path, batch_size):
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

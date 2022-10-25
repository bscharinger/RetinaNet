import tensorflow as tf
from tensorflow import keras
from FeaturePyramid import FeaturePyramid
import numpy as np


def build_head(output_filters, bias_init):
    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init))
        head.add(keras.layers.ReLU())
    head.add(keras.layers.Conv2D(output_filters, 3, 1, padding="same", kernel_initializer=kernel_init,
                                 bias_initializer=bias_init))
    return head


class RetinaNet(keras.Model):
    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes
        prior_prob = tf.constant_initializer(-np.log((1-0.01)/0.01))
        self.cls_head = build_head(9 * num_classes, prior_prob)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_out = []
        box_out = []
        for feature in features:
            box_out.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_out.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))
        cls_out = tf.concat(cls_out, axis=1)
        box_out = tf.concat(box_out, axis=1)
        return tf.concat([box_out, cls_out], axis=-1)


import tensorflow as tf
from tensorflow import keras


def get_backbone():
    backbone = keras.applications.ResNet50(include_top=False, input_shape=[None, None, 3])
    c3_out, c4_out, c5_out = [backbone.get_layer(layer_name).output
                              for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]]
    return keras.Model(inputs=[backbone.inputs], outputs=[c3_out, c4_out, c5_out])


class FeaturePyramid(keras.layers.Layer):
    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_out, c4_out, c5_out = self.backbone(images, training=training)
        p3_out = self.conv_c3_1x1(c3_out)
        p4_out = self.conv_c4_1x1(c4_out)
        p5_out = self.conv_c5_1x1(c5_out)
        p4_out = p4_out + self.upsample_2x(p5_out)
        p3_out = p3_out + self.upsample_2x(p4_out)
        p3_out = self.conv_c3_3x3(p3_out)
        p4_out = self.conv_c4_3x3(p4_out)
        p5_out = self.conv_c5_3x3(p5_out)
        p6_out = self.conv_c6_3x3(c5_out)
        p7_out = self.conv_c7_3x3(tf.nn.relu(p6_out))
        return (p3_out, p4_out, p5_out, p6_out, p7_out)


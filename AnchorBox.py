import tensorflow as tf


class AnchorBox:
    """
    Class for generating anchor boxes that are used to predict object bounding boxes
    """
    def __init__(self):
        self.asp_rat = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1/3, 2/3]]

        self._num_anchors = len(self.asp_rat) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """
        Computes the anchor box dimensions for all ratios and scales at all feature pyramid levels.

        :return: Anchor box dimensions
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.asp_rat:
                anchor_height = tf.math.sqrt(area/ratio)
                anchor_width = area/anchor_height
                dims = tf.reshape(tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2])
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feat_height, feat_width, level):
        """
        Generates the anchor boxes for a given feature map size and level

        :param feat_height: Int representing the height of the feature map
        :param feat_width: Int representing the width of the feature map
        :param level: Int representing the level of the feature map in the feature pyramid
        :return: Anchor boxes with the shape (feature_height * feature width * num_anchors, 4)
        """
        rx = tf.range(feat_width, dtype=tf.float32) + 0.5
        ry = tf.range(feat_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level-3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(self._anchor_dims[level-3], [feat_height, feat_width, 1, 1])
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(anchors, [feat_height * feat_width * self._num_anchors, 4])

    def get_anchors(self, image_height, image_width):
        """
        Generates the anchor boxes for all feature maps of the feature pyramid

        :param image_height: Height of the input image
        :param image_width: Width of the input image
        :return: Anchor boxes for all feature maps stacked as a single tensor with shape (Num_anchors_total, 4)
        """
        anchors = [self._get_anchors(tf.math.ceil(image_height / 2 ** i), tf.math.ceil(image_width / 2 ** i), i, )
                   for i in range(3, 8)]
        return tf.concat(anchors, axis=0)

import tensorflow as tf
import bbox_utils


def ran_flip(image, bboxes):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        bboxes = tf.stack([1-bboxes[:, 2], bboxes[:, 1], 1-bboxes[:, 0], bboxes[:, 3]], axis=-1)
    return image, bboxes


def resize_pad_image(image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0):
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(tf.math.ceil(image_shape/stride)*stride, dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0], padded_image_shape[1])
    return image, image_shape, ratio


def preprocess_sample(sample):
    image = sample["image"]
    bbox = bbox_utils.swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = ran_flip(image, bbox)
    image, image_shape, _ = resize_pad_image(image)

    bbox = tf.stack([bbox[:, 0]*image_shape[1],
                    bbox[:, 1]*image_shape[0],
                    bbox[:, 2]*image_shape[1],
                    bbox[:, 3]*image_shape[0]], axis=-1)
    bbox = bbox_utils.corners2xywh(bbox)
    return image, bbox, class_id

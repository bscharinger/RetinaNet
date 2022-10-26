import tensorflow as tf
import bbox_utils


def ran_flip(image, bboxes):
    """
    Randomly flips an image and its bounding boxes horizontally with a chance of 50%

    :param image: Image as a tensor of shape (height, width, channels)
    :param bboxes: Tensor with shape (num_boxes, 4) containing the bounding boxes
    :return: Randomly flipped image and boxes with the same shapes as the inputs
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        bboxes = tf.stack([1-bboxes[:, 2], bboxes[:, 1], 1-bboxes[:, 0], bboxes[:, 3]], axis=-1)
    return image, bboxes


def resize_pad_image(image, min_side=800.0, max_side=1333.0, jitter=None, stride=128.0):
    """
    Resizes and pads an image while preserving the aspect ratio.
    1. Resize image so that the shorter side is equal to <min_side>
    2. If the longer side is greater than <max_side> resize image so that the longer side is equal to <max_side>
    3. Pad the image with zeros on the right and bottom to make the image shape divisable by <stride>

    :param image: Image as a tensor of shape (height, width, channels)
    :param min_side: Shorter side of the image is resized to this value if <jitter> is None
    :param max_side: If longer side of the image exceeds this value after resizing, image is resized so that the
                     longer side equals <max_side>
    :param jitter: A list of floats containing the minimum and maximum size for scale jittering. If available,
                   the shorter side of the image will be resized to a random value in this range
    :param stride: Stride of the smallest feature map in the feature pyramid.
    :return: image: Resized and padded image
             image_shape: Shape of the image before padding
             ratio: Scaling factor used to resize the image
    """
    if jitter is None:
        jitter = [640, 1024]
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
    """
    Applies all preprocessing steps to a single training sample
    :param sample: Dict containing a single training sample
    :return: image: Resized and padded image with random horizontal flipping
             bbox: Bounding boxes with shape (num_boxes, 4)
             class_id: Tensor representing the class id of the objects with shape (num_objects,)
    """
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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def swap_xy(boxes):
    return tf.stack([boxes[:,1], boxes[:,0], boxes[:,3], boxes[:,2]], axis=-1)


def corners2xywh(boxes):
    return tf.concat([(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]], axis=-1)


def xywh2corners(boxes):
    return tf.concat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)

def calc_iou(x, y):
    x_corners = xywh2corners(x)
    y_corners = xywh2corners(y)
    lu = tf.maximum(x_corners[:, None, :2], y_corners[:, :2])
    rd = tf.minimum(x_corners[:, None, 2:], y_corners[:, 2:])
    intersection = tf.maximum(0.0, rd-lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    x_area = x[:, 2] * x[:, 3]
    y_area = y[:, 2] * y[:, 3]
    union = tf.maximum(x_area[:, None] + y_area - intersection_area, 1e-8)
    return tf.clip_by_value(intersection_area / union, 0.0, 1.0)

def visualize_bbox(image, bboxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]):
    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(bboxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(patch)
        ax.text(x1, y1, text, bbox={"facecolor": color, "alpha": 0.4}, clip_box=ax.clipbox, clip_on=True)
    plt.show()
    return ax

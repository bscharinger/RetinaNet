import AnchorBox
import tensorflow as tf
import bbox_utils


class LabelEncoder:
    """
    Class to transform the raw labels into targets for training
    """

    def __init__(self):
        self._anchor_box = AnchorBox.AnchorBox()
        self._box_var = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    @staticmethod
    def _match_anchor_boxes(anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4):
        """
        Matches the ground truth boxes to anchor boxes based in IOU
        1. Calculate pairwise IOU for M anchor boxes and N ground truth bounding boxes
        2. Assign ground truth bounding box  with maximum IOU in each row to anchor its anchor box
           provided the IOU is greater than <match_iou>
        3. If the maximum IOU in a row is less than <match_iou> the anchor box is assigned with the background class
        4. Remaining anchor boxes are ignored during training

        :param anchor_boxes: Tensor with the shape (Num_anchors_total, 4) containing all anchor boxes for
                             given image shape
        :param gt_boxes: Tensor with the shape (num_objects, 4) containing the ground truth bounding boxes
                         in format (x_center, y_center, width, height)
        :param match_iou: Float representing the minimum IOU for assigning GT bboxes to anchor boxes
        :param ignore_iou: Float representing IOU threshold, under which an anchor box is assigned to background
        :return: matched_gt_idx: Index of matched object
                 positive_mask: Mask for anchor boxes that are assigned ground truth boxes
                 ignore_mask: Mask for anchor boxes that are ignored during training
        """
        iou_mat = bbox_utils.calc_iou(anchor_boxes, gt_boxes)
        max_iou = tf.reduce_max(iou_mat, axis=1)
        matched_gt_idx = tf.argmax(iou_mat, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return matched_gt_idx, tf.cast(positive_mask, dtype=tf.float32), tf.cast(ignore_mask, dtype=tf.float32)

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """
        Transforms the ground truth boxes into targets for training

        :param anchor_boxes: Anchor boxes
        :param matched_gt_boxes: Matched GT boxes
        :return: Training targets
        """
        box_target = tf.concat([(matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:])], axis=-1)
        box_target = box_target / self._box_var
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """
        Creates box and classification targets for a single sample

        :param image_shape: Image shape
        :param gt_boxes: Ground truth boxes
        :param cls_ids: Class IDs
        :return: Label
        """
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_boxes)
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids)
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)
        return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """
        Encodes box and classification targets for a batch
        :param batch_images:
        :param gt_boxes:
        :param cls_ids:
        :return:
        """
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()


class DecodePredictions(tf.keras.layers.Layer):
    """
    Keras Layer to decode predictions of the RetinaNet model
    """
    def __init__(self, num_classes=80, confidence_threshold=0.05, nms_iou_threshold=0.5, max_detections_per_class=100,
                 max_detections=100, box_variance=None, **kwargs):
        """

        :param num_classes: Number of classes in the Dataset
        :param confidence_threshold: Minimum class probability
        :param nms_iou_threshold: IOU threshold for NMS operation
        :param max_detections_per_class: Maximum number of detections to retain per class
        :param max_detections: Maximum number of detections to retain across all classes
        :param box_variance: Scaling factors used to scale the bbox predictions
        """
        super(DecodePredictions, self).__init__(**kwargs)
        if box_variance is None:
            box_variance = [0.1, 0.1, 0.2, 0.2]
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections
        self._anchor_box = AnchorBox.AnchorBox()
        self._box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat([boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                           tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:]], axis=-1)
        boxes_transformed = bbox_utils.xywh2corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        return tf.image.combined_non_max_suppression(tf.expand_dims(boxes, axis=2),
                                                     cls_predictions, self.max_detections_per_class,
                                                     self.max_detections, self.nms_iou_threshold,
                                                     self.confidence_threshold, clip_boxes=False)

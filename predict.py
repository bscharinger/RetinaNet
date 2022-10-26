import tensorflow as tf
from tensorflow import keras
import LabelCoding
import FeaturePyramid
import RetinaNet
import Losses
import tensorflow_datasets as tfds
import bbox_utils


def prep_img(img):
    img, _, rat = preprocessing.resize_pad_image(img, jitter=None)
    img = tf.keras.applications.resnet.preprocess_input(img)
    return tf.expand_dims(img, axis=0), rat


# initializing model and loading weights
import preprocessing

model_dir = "E:\\ML_projects\\RetinaNet\\models"

weights_dir = model_dir
label_encoder = LabelCoding.LabelEncoder()

num_classes = 80
batch_size = 2

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=learning_rate_boundaries,
                                                                  values=learning_rates)

resnet50_backbone = FeaturePyramid.get_backbone()
loss_fn = Losses.RetinaNetLoss(num_classes)
model = RetinaNet.RetinaNet(num_classes, resnet50_backbone)

latest_ckpt = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_ckpt)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = LabelCoding.DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = keras.Model(inputs=image, outputs=detections)

test_ds, ds_info = tfds.load("coco/2017", split="test", data_dir="E:\ML_projects\RetinaNet\data", with_info=True)
int2str = ds_info.features["objects"]["label"].int2str

test_ds = test_ds.shuffle(20000)

for sample in test_ds.take(16):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prep_img(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]]
    bbox_utils.visualize_bbox(image, detections.nmsed_boxes[0][:num_detections]/ratio,
                              class_names, detections.nmsed_scores[0][:num_detections])






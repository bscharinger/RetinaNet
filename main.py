import tensorflow as tf
from tensorflow import keras
import LabelCoding
import RetinaNet
import FeaturePyramid
import Losses
import os
import tensorflow_datasets as tfds
import preprocessing
import bbox_utils


def prep_img(img):
    img, _, rat = preprocessing.resize_pad_image(img, jitter=None)
    img = tf.keras.applications.resnet.preprocess_input(img)
    return tf.expand_dims(img, axis=0), rat


model_dir = "E:\\ML_projects\\RetinaNet\\models"
label_encoder = LabelCoding.LabelEncoder()

num_classes = 80
batch_size = 1

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=learning_rate_boundaries,
                                                                  values=learning_rates)

resnet50_backbone = FeaturePyramid.get_backbone()
loss_fn = Losses.RetinaNetLoss(num_classes)
model = RetinaNet.RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(model_dir, "weights"+"_epoch_{epoch}"),
                                             monitor="loss", save_best_only=False, save_weights_only=True, verbose=1)]
(train_ds, val_ds, test_ds), dataset_info = tfds.load("coco/2017",
                                                      split=["train", "validation", "test"],
                                                      with_info=True,
                                                      data_dir="E:\\ML_projects\\RetinaNet\\data",
                                                      download=False)

autotune = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocessing.preprocess_sample, num_parallel_calls=autotune)
train_ds = train_ds.shuffle(2000, reshuffle_each_iteration=True)
train_ds = train_ds.padded_batch(batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
train_ds = train_ds.map(label_encoder.encode_batch, num_parallel_calls=autotune)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
train_ds = train_ds.prefetch(autotune)

val_ds = val_ds.map(preprocessing.preprocess_sample, num_parallel_calls=autotune)
val_ds = val_ds.padded_batch(batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
val_ds = val_ds.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.prefetch(autotune)

# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // batch_size
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // batch_size
#
# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch
epochs = 20


# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

latest_ckpt = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_ckpt)

model.fit(train_ds.take(20000), validation_data=val_ds.take(2000), epochs=epochs, callbacks=callbacks, verbose=1)

latest_ckpt = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_ckpt)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = LabelCoding.DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = keras.Model(inputs=image, outputs=detections)

int2str = dataset_info.features["objects"]["label"].int2str


for sample in test_ds.take(5):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prep_img(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]]
    bbox_utils.visualize_bbox(image, detections.nmsed_boxes[0][:num_detections]/ratio,
                              class_names, detections.nmsed_scores[0][:num_detections])


# RetinaNet
This is a keras implementation of object Detection with RetinaNet trained on the COCO 2017 Dataset. RetinaNet is a single-stage object detector using a feature pyramid network on top of a ResNet50 backbone and uses Focal loss and Smooth L1 loss.

Retina Net was introduced by Lin et. al., their paper can be found [here](https://arxiv.org/abs/1708.02002)

This is a work in progress and will be updated regularily.

## Training

Training is done using the [main.py](main.py) file. To change the amount of images used for training and testing, modify the `tf_dataset.take()` statements in the `model.fit()` function in [main.py](main.py). 

The network was trained on a PC with a RTX 2070 with 8GB of memory. For now the network was trained for 28 Epochs with up to 40.000 images. Since training takes extremely long, the network improves only slowly, better results are expected with more training.

## Results
For inference, run the file [predict.py](predict.py).

For now, the network delivers relatively good results for objects it knows already, but it has still a lot of objects to learn.


![Train_people](https://user-images.githubusercontent.com/105383316/199739419-416eb428-f2c5-4e34-a79d-3d9a44579f9e.png)
![table](https://user-images.githubusercontent.com/105383316/199739459-e6c085c4-4dd5-4313-b234-05d7bc4daf77.png)
![elephant](https://user-images.githubusercontent.com/105383316/199739488-c33b254b-69c7-4ffa-8559-094f49fe1c06.png)



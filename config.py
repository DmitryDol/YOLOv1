import os
import torchvision as tv

DATA_PATH = 'data'
CLASSES_PATH = os.path.join(DATA_PATH, 'classes.json')

BATCH_SIZE = 64
EPOCHS = 135
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-4

EPSILON = 1E-6
IMG_SIZE = (448, 448)


# constants from paper
S = 7       # Divide each image into a SxS grid
B = 2       # Number of bounding boxes to predict
C = 20      # Number of classes in the dataset
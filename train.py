from models.ACGAN import ACGAN

from keras.datasets import mnist
IMG_SIZE = 28
CHANNELS = 1
CLASSES = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train)

acgan = ACGAN(IMG_SIZE,IMG_SIZE,CHANNELS,CLASSES)

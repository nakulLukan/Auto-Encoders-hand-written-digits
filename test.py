import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, ReLU
from tqdm import tqdm

(x_train, Y), (x_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_test_dataset = x_test/255.
CHECKPOINT_PATH = './checkpoints_adam_mse_sigmoid_28x7x28_F_2_3/checkpoint.ckpt'

model = tf.keras.models.load_model(CHECKPOINT_PATH)

for test_data in X_test_dataset:
    X = np.array(test_data).reshape(-1,28,28,1)

    prediction = model.predict(X)
    output_image = prediction[0].reshape(28,28)
    plt.imshow(output_image, cmap='gray')
    plt.show()
    plt.imshow(test_data, cmap='gray')
    plt.show()





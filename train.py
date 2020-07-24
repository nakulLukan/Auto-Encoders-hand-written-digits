import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, LeakyReLU, MaxPooling2D, Dropout, Conv2DTranspose, Reshape, ReLU
from tqdm import tqdm

(x_train, Y), (x_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train_dataset = x_train/255.

def create_model():
    model = Sequential()

    model.add(Conv2D(2, (3,3), strides=(2,2), input_shape=(28,28,1), padding='same', activation='relu'))

    model.add(Conv2D(3, (3,3), strides=(2,2), padding='same',  activation='relu'))
    
    model.add(Flatten())
    model.add(Dense(7*7*1))

    model.add(Reshape((7,7,1)))
    model.add(Conv2DTranspose(3,(3,3), strides=(2,2),padding='same', activation='relu'))

    model.add(Conv2DTranspose(2,(3,3), strides=(2,2),padding='same', activation='relu'))
    
    model.add(Conv2DTranspose(1,(3,3), strides=(1,1),padding='same', activation='sigmoid'))
    assert model.output_shape == (None, 28,28,1)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

model = create_model()

mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

EPOCHS = 15
CHECKPOINT_PATH = './checkpoints_adam_mse_sigmoid_28x7x28_F_2_3/checkpoint.ckpt'

X = np.array(X_train_dataset).reshape(-1,28,28,1)

for episode in tqdm(range(0,EPOCHS), ascii=True, unit='episode'):
    model.fit(X, X, batch_size=64)
    if episode!=0:
        model.save(filepath=CHECKPOINT_PATH)

print("training completed")
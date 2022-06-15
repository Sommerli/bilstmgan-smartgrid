# %%
from __future__ import print_function, division

import datetime
import pickle

from keras import Sequential, Input, Model
from keras.layers import Bidirectional, Dropout, LSTM, RepeatVector, TimeDistributed, LeakyReLU, Dense
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

from train_utils import *
from datetime import datetime

data_path = './datasets/'
model_path = './models/'


def load_data():
    # with open(data_path + 'dataset_one_week_not_shuffled.pkl', 'rb') as f:
    with open(data_path + 'dataset_one_month_not_shuffled.pkl','rb') as f:
        dataset = pickle.load(f)
    x_train, y_train, x_test, y_test = dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']
    return x_train, y_train, x_test, y_test


def run_and_plot(learning_rate, batch_size, epochs, save_interval=50):
    x_train, y_train, x_test, y_test = load_data()
    gan = LSTMGAN(epochs, learning_rate)
    discriminator_loss, gan_loss = gan.train(epochs, batch_size, save_interval)  # x_train, batch_size, epochs, generator, discriminator, gan, progress)
    y_pred = anomaly_detection(x_test, y_test, batch_size, gan.discriminator)
    print(len(y_pred), len(y_test))
    acc_score, precision, recall, f1 = evaluation(y_test, y_pred)
    print(f'{acc_score=}, {precision=}, {recall=}, {f1=}')
    plt.figure(figsize=(20, 20))

    plt.subplot(2, 1, 1)
    plt.plot(discriminator_loss, label='Discriminator')
    plt.plot(gan_loss, label='Generator')
    plt.title("Training Losses")
    plt.legend()

    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    plt.subplot(2, 2, 4)
    plot_roc_curve(y_test, y_pred)

    plt.show()

    return y_pred, gan


def build_generator():
    model = Sequential()
    # encoder
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(8, 1)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.2))
    # specifying output to have 8 timesteps
    model.add(RepeatVector(8))
    # decoder
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(256)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(1)))
    model.add(LeakyReLU(alpha=0.2))
    model.summary()

    noise = Input(shape=(8, 1))
    img = model(noise)

    return Model(noise, img, name='Generator')


def build_discriminator():
    model = Sequential()

    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(8, 1)))
    model.add(Dropout(0.2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Bidirectional(LSTM(256)))
    model.add(Dropout(0.2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(RepeatVector(1))
    model.add(TimeDistributed(Dense(300)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(300)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(300)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(1)))
    model.summary()

    img = Input(shape=(8, 1))
    validity = model(img)

    return Model(img, validity, name='Discriminator')


class LSTMGAN:
    def __init__(self, epochs=1000, learning_rate=0.00001, import_model=False):
        self.epochs = epochs
        self.batch_size = 0

        # Performance metrics
        self.generated_data = [[] for _ in range(self.epochs)]
        self.training_data = [[] for _ in range(self.epochs)]
        self.generated_average = []
        self.training_average = []

        optimizer = Adam(learning_rate)  # , 0.4)

        # Build and compile the discriminator
        self.discriminator = build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = build_generator()

        # The generator takes noise as input and generates samples
        z = Input(shape=(8, 1))
        sample = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated samples as input and determines validity
        valid = self.discriminator(sample)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='mean_squared_error', optimizer=optimizer)

    def train(self, epochs, batch_size=128, save_interval=10):
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size

        # Load the dataset
        X_train = load_data()[0]

        # Rescale -1 to 1
        # X_train = X_train / 127

        batch_count = X_train.shape[0] // batch_size

        # Adversarial ground truths
        valid = np.ones((batch_size, 1, 1))
        fake = np.zeros((batch_size, 1, 1))

        g_loss_epochs = np.zeros((self.epochs, 1))
        d_loss_epochs = np.zeros((self.epochs, 1))

        for epoch in range(self.epochs):
            for index in range(batch_count):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a batch of data
                training_batch = X_train[index * batch_size: (index + 1) * batch_size]
                self.training_data[epoch].extend(training_batch.flatten())

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, 8, 1))
                generated_batch = self.generator.predict(noise)
                self.generated_data[epoch].extend(generated_batch.flatten())

                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss_real = self.discriminator.train_on_batch(training_batch, valid)
                d_loss_fake = self.discriminator.train_on_batch(generated_batch, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, valid)

                # save loss history
                g_loss_epochs[epoch] = g_loss
                d_loss_epochs[epoch] = d_loss[0]

            # Plot the progress
            self.generated_average.append(pd.Series(self.generated_data[epoch]).rolling(window=50).mean())
            self.training_average.append(pd.Series(self.training_data[epoch]).rolling(window=50).mean())

            print(datetime.now().strftime() + "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        return g_loss_epochs, d_loss_epochs


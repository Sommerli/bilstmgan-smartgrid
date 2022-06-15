from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
import pickle

from keras import Sequential, Input, Model
from keras.layers import Bidirectional, Dropout, LSTM, RepeatVector, TimeDistributed, LeakyReLU, Dense
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from keras.models import load_model

import datetime
import time

from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_path = './datasets/'
model_path = './models/'

def anomaly_detection(x_test, y_test, batch_size, discriminator):
    nr_batches_test = np.ceil(x_test.shape[0] // batch_size).astype(np.int32)

    results = []

    for t in range(nr_batches_test + 1):
        ran_from = t * batch_size
        ran_to = (t + 1) * batch_size
        image_batch = x_test[ran_from:ran_to]
        tmp_rslt = discriminator.predict(
            x=image_batch, batch_size=128, verbose=0)
        results = np.append(results, tmp_rslt)

    pd.options.display.float_format = '{:20,.7f}'.format
    results_df = pd.concat(
        [pd.DataFrame(results), pd.DataFrame(y_test)], axis=1)
    results_df.columns = ['results', 'y_test']
    # print ('Mean score for normal packets :', results_df.loc[results_df['y_test'] == 0, 'results'].mean() )
    # print ('Mean score for anomalous packets :', results_df.loc[results_df['y_test'] == 1, 'results'].mean())

    # Obtaining the lowest 3% score
    per = np.percentile(results, 3)
    y_pred = results.copy()
    y_pred = np.array(y_pred)

    # Thresholding based on the score
    inds = (y_pred > per)
    inds_comp = (y_pred <= per)
    y_pred[inds] = 0
    y_pred[inds_comp] = 1

    return y_pred, results_df

def load_data():
    with open(data_path + 'dataset_one_month_not_shuffled.pkl',
              'rb') as f:  # 'dataset_one_week_not_shuffled.pkl', 'rb') as f:
        dataset = pickle.load(f)
    x_train, y_train, x_test, y_test = dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']
    return x_train, y_train, x_test, y_test

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['normal', 'anomaly']
    # plt.figure(figsize=(10,10),)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc_curve(y_test, y_pred):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    # plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras,
             label='Keras (area = {:.2f})'.format(auc_keras))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def run_and_plot(dataset, learning_rate, batch_size, epochs, save_interval=50):
    x_train, y_train, x_test, y_test = dataset['x_train'], dataset[
        'y_train'], dataset['x_test'], dataset['y_test']
    gan = LSTMGAN(epochs, learning_rate)
    discriminator_loss, gan_loss = gan.train(epochs, batch_size, save_interval)  # x_train, batch_size, epochs, generator, discriminator, gan, progress)
    y_pred = anomaly_detection(x_test, y_test, batch_size, gan.discriminator)
    print(len(y_pred), len(y_test))
    # acc_score, precision, recall, f1 = evaluation(y_test, y_pred)

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
    # model.summary()

    noise = Input(shape=(8, 1))
    img = model(noise)

    return Model(noise, img)


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
    # model.summary()

    img = Input(shape=(8, 1))
    validity = model(img)

    return Model(img, validity)

class LSTMGAN:
    def __init__(self, learning_rate=0.00001):

        optimizer = Adam(learning_rate)  # , 0.4)

        # Build and compile the discriminator
        self.discriminator = build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build or load the generator
        self.generator = build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(8, 1))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='mean_squared_error', optimizer=optimizer)

    def train(self, epochs, batch_size=128, save_interval=10):
        # Load the dataset
        X_train = load_data()[0]

        # Rescale -1 to 1
        # X_train = X_train / 127

        batch_count = X_train.shape[0] // batch_size

        # Adversarial ground truths
        valid = np.ones((batch_size, 1, 1))
        fake = np.zeros((batch_size, 1, 1))

        g_loss_epochs = np.zeros((epochs, 1))
        d_loss_epochs = np.zeros((epochs, 1))

        for epoch in range(epochs):
            for index in range(batch_count):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a batch of data
                training_batch = X_train[index * batch_size: (index + 1) * batch_size]

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, 8, 1))
                generated_batch = self.generator.predict(noise)

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
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        return g_loss_epochs, d_loss_epochs


learning_rate = 0.00001
epochs = 10
batch_size = 512
lstmgan = LSTMGAN(learning_rate)

start_time = time.perf_counter_ns()
g_loss, d_loss = lstmgan.train(epochs=epochs, batch_size=batch_size, save_interval=10)
end_time = time.perf_counter_ns()
time_formatted = str(datetime.timedelta(seconds=(end_time - start_time) * 10**(-9))) # gives time in hh:mm:ss

print(f"Trained the model in {time_formatted}")

for i in range(0, lstmgan.epochs, 9):
    y_gen = lstmgan.generated_data[i][-1000:]
    x_gen = range(len(y_gen))

    y = lstmgan.training_data[i][-1000:]
    x = range(len(y))

    fig = plt.figure(figsize=(8,3))
    plt.plot(x, y, 'b', label='Training data', alpha=0.5)
    plt.plot(x_gen, y_gen, 'r', label='Generated data', alpha=0.5)

    plt.xlabel('Timestep')
    plt.ylabel('Normalized load')
    plt.title(f'{i} epochs')
    plt.legend()
    plt.show()
    fig.savefig(f'{i}_epochs_train_vs_gen.png')

for i in range(0, lstmgan.epochs, 9):

    y_gen = lstmgan.generated_average[i]
    x_gen = range(len(y_gen))

    y = lstmgan.training_average[i]
    x = range(len(y))

    fig = plt.figure(figsize=(10,3))
    plt.plot(x, y, 'b', label='Training data avg', alpha=0.5)
    plt.plot(x_gen, y_gen, 'r', label='Generated data avg', alpha=0.5)
    plt.title(f'{i} epochs')
    plt.legend()
    plt.show()
    fig.savefig(f'{i}_epochs_train_vs_gen_averages.png')

_, y_train, x_test, y_test = load_data()
start_time = time.perf_counter_ns()
y_pred, result_df = anomaly_detection(x_test, y_test, lstmgan.batch_size, lstmgan.discriminator)
end_time = time.perf_counter_ns()
time_formatted = str(datetime.timedelta(seconds=(end_time - start_time) * 10**(-9))) # gives time in hh:mm:ss

print(f"Made predictions in {time_formatted}")
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)
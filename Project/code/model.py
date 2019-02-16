from __future__ import print_function

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau, CSVLogger

import os.path
import time
import pandas as pd

options = {
    "checkpoints_path": "./checkpoints",
    "csv_logs_path": "./logs",
    "figures_path": "./figures",
    "summary_path": "./summary"
}


class TimerCallback(Callback):
    """ Custom made callback to measure epoch time. """

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        epoch_time = time.time() - self.epoch_time_start
        self.times.append(epoch_time)


class ResetStateCallback(Callback):
    """ A callback to reset LSTM states at the start of each epoch (stateful=True). """

    def on_epoch_begin(self, batch, logs={}):
        print("Resetting LSTM/GRU states...")
        self.model.reset_states()


def build_model(network_type, neurons, dropout, optimizer, batch_size, timesteps,
                horizon, checkpoint_name, stateful=False, seq2seq=False, plot=False):
    """ Builds and returns the model. """

    # In case the network is stateless let batch_size = None to handle variable lengths
    if not stateful:
        batch_size = None

    # In case we are interested in a seq2seq then the horizon is set to 1
    if seq2seq:
        horizon = 1

    # Number of layers is defined by the number of integers in the neurons list
    layers = len(neurons)

    if network_type == "LSTM":
        model = Sequential()
        for i in range(layers):
            # In case the model is framed as a seq2seq we need return_sequences to all layers
            if not seq2seq:
                return_sequences = False if i == (layers - 1) else True
            else:
                return_sequences = True
            model.add(LSTM(neurons[i],
                           input_shape=(timesteps, 1),
                           batch_size=batch_size,
                           return_sequences=return_sequences,
                           stateful=stateful))
            model.add(Dropout(dropout))

        model.add(Dense(horizon, activation='linear'))
        model.compile(loss='mse', optimizer=optimizer)
        model.summary()

    elif network_type == "GRU":
        model = Sequential()
        for i in range(layers):
            # In case the model is framed as a seq2seq we need return_sequences to all layers
            if not seq2seq:
                return_sequences = False if i == (layers - 1) else True
            else:
                return_sequences = True
            model.add(GRU(neurons[i],
                          input_shape=(timesteps, 1),
                          batch_size=batch_size,
                          return_sequences=return_sequences,
                          stateful=stateful))
            model.add(Dropout(dropout))

        model.add(Dense(horizon, activation='linear'))
        model.compile(loss='mse', optimizer=optimizer)
        model.summary()

    else:
        raise TypeError("Wrong network_type={} argument.Valid options are (LSTM, GRU)".format(network_type))

    if plot:
        model_to_png(model, checkpoint_name)

    # Prints model summary to a file
    model_summary_to_file(model, checkpoint_name)

    print("Model stateful:", stateful)
    print("Optimizer:", optimizer)
    print("Seq2seq:", seq2seq)

    return model


def get_callbacks(checkpoint_name, stateful=False):
    """ Returns a list with callbacks used in training.
        - EarlyStopping: to not overfit and stop training when there is no imporovement in the monitored quantity.
        - ModelCheckpoint: to save the best model when appropriate.
        In case the network is stateful:
        - ResetStateCallback: to reset states after each epoch.
    """

    model_path = "/".join([options.get("checkpoints_path"), checkpoint_name])
    log_path = "/".join([options.get("csv_logs_path"), checkpoint_name.split(".h5")[0] + ".csv"])

    print(log_path)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, verbose=1, min_delta=0, mode="auto"),
        ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True,
                        save_weights_only=False, mode="auto"),
        ReduceLROnPlateau(monitor="val_loss", patience=5, verbose=1, factor=0.5, min_lr=1e-4),
        CSVLogger(filename=log_path, separator=',', append=True)
    ]

    if stateful:
        callbacks.append(ResetStateCallback())

    return callbacks


def model_exists(checkpoint_name):
    """ Checks whether a trained model exists and returns True/False. """

    path = "/".join([options.get("checkpoints_path"), checkpoint_name])
    if os.path.exists(path):
        return True

    return False


def get_model(checkpoint_name):
    """ Loads and returns the trained model. """

    path = "/".join([options.get("checkpoints_path"), checkpoint_name])
    return load_model(path)


def save_model(model, checkpoint_name):
    """ Saves the given trained model to a file specified by checkpoints_path. """
    path = "/".join([options.get("checkpoints_path"), checkpoint_name])
    model.save(path)


def model_to_png(model, checkpoint_name):
    """ Plots the model to a picture. """

    checkpoint_name = checkpoint_name.split(".h5")[0] + ".png"
    path = "/".join([options.get("figures_path"), checkpoint_name])
    plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)


def model_summary_to_file(model, checkpoint_name):
    """ Prints the summary of the model in a file. """

    checkpoint_name = checkpoint_name.split(".h5")[0] + ".txt"
    path = "/".join([options.get("summary_path"), checkpoint_name])
    with open(path, 'w+') as fd:
        model.summary(print_fn=lambda x: fd.write(x + "\n"))


def read_log(checkpoint_name):
    """ Reads the log and returns a dictionary. """

    cols = ["loss", "val_loss"]
    log_path = "/".join([options.get("csv_logs_path"), checkpoint_name.split(".h5")[0] + ".csv"])
    df = pd.read_csv(log_path, usecols=cols)

    return df.to_dict(orient='list')

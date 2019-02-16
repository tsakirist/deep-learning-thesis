from __future__ import print_function
from __future__ import division

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
from math import sqrt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

# Use ggplot style-sheet
# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')

# # Set user defined parameters for pyplot
# params = {
#     "figure.titlesize": 18,
#     "figure.figsize": (14, 9),
#     "figure.facecolor": "white",
#     "axes.labelsize": 16,
#     "axes.facecolor": "white",
#     "lines.linewidth": 1.5,
#     "legend.shadow": True,
#     "legend.loc": "best",
# }
# plt.rcParams.update(params)

plot_latex_dir = "./plots_latex"
model_name = ""


def load_dataset(file, plot=False):
    """ Loads the dataset from CSV and returns a dataframe. """

    path = os.path.dirname(__file__)
    path = os.path.join(path, "../dataset/", file)
    path = os.path.abspath(path)
    print("Filepath:\n", path, sep='')

    # Custom made date_parser for flexibility
    # date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    # date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m")
    # date_parser = lambda x: pd.datetime.strptime('190' + x, '%Y-%m')
    # series = pd.read_csv(path, parse_dates=[0], date_parser=date_parser,
    #                      header=0, index_col=0, skipfooter=1, engine='python')

    # Appropriately reading csv according to different datasets
    if "beijing" in file:
        series = pd.read_csv(path, header=0, index_col=0)
        series.set_index(pd.to_datetime(series[['year', 'month', 'day', 'hour']]), inplace=True)
        series = series[["pm2.5"]]
        series.dropna(inplace=True)
        # series = series[['PRES']]
        # series.insert(0, 'datetime', pd.to_datetime(series[['year', 'month', 'day', 'hour']]))
        # series.drop(series.columns[1:6], axis=1, inplace=True)

    elif "zurich" in file:
        date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m")
        series = pd.read_csv(path, parse_dates=[0], date_parser=date_parser,
                             header=0, index_col=0, skipfooter=1, engine='python')

    elif "airline" in file:
        date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m")
        series = pd.read_csv(path, parse_dates=[0], date_parser=date_parser,
                             header=0, index_col=0, skipfooter=1, engine='python')

    else:
        date_parser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        series = pd.read_csv(path, parse_dates=[0], date_parser=date_parser,
                             header=0, index_col=0, skipfooter=1, engine='python')

    # series = series.astype('float32')

    # Plot the series
    if plot:
        plt.title(file.split(".")[0])
        plt.plot(series)
        plt.show()

    return series


def split_dataset(series, test_ratio=0.2, val_ratio=0.1):
    """ Splits the dataset into training, validation and test set. """

    # Split into training and test set
    test_pos = int(series.shape[0] * (1 - test_ratio))
    values = series.values
    train, test = values[:test_pos], values[test_pos:]

    # Split the training set into a validation aswell
    val_pos = int(train.shape[0] * (1 - val_ratio))
    train, validation = train[:val_pos], train[val_pos:]

    return train, validation, test


def convert_to_supervised(series, lookback, lookahead=1):
    """ Converts the data into supervised learning according to the lookback and lookahead.
        i.e. Make the input-sequence of the series to input-output pairs [X, y].
    """

    df = pd.DataFrame(series)
    cols = [df.shift(-i) for i in range(1, lookback + lookahead)]
    cols.insert(0, df)
    df = pd.concat(cols, axis=1)
    df.dropna(inplace=True)
    cols_names = ["t-" + str(i) for i in range(lookback, 0, -1)]
    cols_names += ["t"]
    if lookahead > 1:
        cols_names += ["t+" + str(i) for i in range(1, lookahead)]
    df.columns = cols_names

    # print(df.head(5))
    return df


def difference_series(series):
    """ Applies the first order difference of the series to make it stationary. """

    series = series.diff()
    series.dropna(inplace=True)

    return series


def undifference_series(series, first_value):
    """ Applies the reverse operation of the difference operator to get the original series. """

    series.iloc[0] += first_value
    series = series.cumsum()

    return series


def normalize_series(train, validation, test):
    """ Returns the normalized train, test series.
    Normalization transforms the series in the range of (0, 1) or (-1, 1).
    """

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_validation = scaler.transform(validation)
    scaled_test = scaler.transform(test)

    return scaler, scaled_train, scaled_validation, scaled_test


def standardize_series(train, validation, test):
    """ Returns the standardized train, test series.
    Standardization makes the series have zero mean and unit variance.
    """

    std_scaler = StandardScaler()
    std_scaler = std_scaler.fit(train)
    std_train = std_scaler.transform(train)
    std_validation = std_scaler.transform(validation)
    std_test = std_scaler.transform(test)

    return std_scaler, std_train, std_validation, std_test


def invert_scale_series(scaler, series):
    """ De-normalizes the given series according to the scaler. """

    return scaler.inverse_transform(series)


def invert_scale_single(scaler, value):
    """
    Inverts the scale for a forecasted value.
    """

    return scaler.inverse_transform(value)


def truncate(train, validation, test, batch_size):
    """ Truncates the train, validation and test datasets to be divisors of the batch_size. """

    train_size = (train.shape[0] // batch_size) * batch_size
    val_size = (validation.shape[0] // batch_size) * batch_size
    test_size = (test.shape[0] // batch_size) * batch_size
    train, validation, test = train[:train_size], validation[:val_size], test[:test_size]

    return train, validation, test


def preprocess_data(data, lookback, lookahead, stateful, batch_size, test_ratio=0.2, val_ratio=0.1):
    """ Data pre-processing:
        -> Training-Test split
        -> Normalization
        -> Time-Series to Supervised Learning
    Returns scaler, train, test.
    """

    train, validation, test = split_dataset(data, test_ratio, val_ratio)
    scaler, scaled_train, scaled_validation, scaled_test = normalize_series(train, validation, test)
    # scaler, scaled_train, scaled_validation, scaled_test = standardize_series(train, validation, test)
    supervised_train = convert_to_supervised(scaled_train, lookback, lookahead)
    supervised_validation = convert_to_supervised(scaled_validation, lookback, lookahead)
    supervised_test = convert_to_supervised(scaled_test, lookback, lookahead)

    # If model is stateful we need to truncate datasets in order to make them divisor of batch_size
    if stateful:
        supervised_train, supervised_validation, supervised_test = truncate(supervised_train,
                                                                            supervised_validation,
                                                                            supervised_test,
                                                                            batch_size)

    return scaler, supervised_train, supervised_validation, supervised_test


def mse(true, predictions):
    """ Returns the Mean-Square-Error between the actual and predicted values. """

    return MSE(true, predictions)


def rmse(true, predictions):
    """ Returns the Root-Mean-Square-Error between the actual and predicted values. """

    return sqrt(MSE(true, predictions))


def mae(true, predictions):
    """ Returns the Mean-Absolute-Error between the actual and predicted values. """

    return MAE(true, predictions)


def mape(true, predictions):
    """ Returns the Mean-Absolute-Percentage-Error between the actual and predicted values. """

    true, predictions = np.array(true), np.array(predictions)

    return np.mean(np.abs((true - predictions) / true)) * 100


def smape(true, predictions):
    """ Returns the Symmetric-Mean-Absolute-Percentage-Error between the actual and predicted values. """

    true, predictions = np.array(true), np.array(predictions)

    return np.mean(np.abs(true - predictions) / (np.abs(true) + np.abs(predictions))) * 200


def print_statistic_metrics(train_true, train_pred, test_true, test_pred):
    """ Prints statistic metrics for the training & test datasets. """

    train_rmse = rmse(train_true, train_pred)
    train_mae = mae(train_true, train_pred)
    train_mape = mape(train_true, train_pred)
    train_smape = smape(train_true, train_pred)

    test_rmse = rmse(test_true, test_pred)
    test_mae = mae(test_true, test_pred)
    test_mape = mape(test_true, test_pred)
    test_smape = smape(test_true, test_pred)

    print("-------------------- Accuracy Statistics --------------------")
    print("Train-RMSE: {:.3f}\nTest-RMSE:  {:.3f}".format(train_rmse, test_rmse))
    print("Train-MAE:  {:.3f}\nTest-MAE:   {:.3f}".format(train_mae, test_mae))
    print("Train-MAPE: {:.3f}%\nTest-MAPE:  {:.3f}%".format(train_mape, test_mape))
    print("Train-SMAPE:{:.3f}%\nTest-SMAPE: {:.3f}%".format(train_smape, test_smape))
    print("-------------------- -------------------- --------------------")


def gcd(a, b):
    """ Computes the gcd of two numbers. """

    if b == 0:
        return a

    return gcd(b, a % b)


def get_optimal_batch_size(train_size, test_size):
    """ Returns the optimum batch size by finding the highest common factor. """

    batch_size = gcd(train_size, test_size)

    return batch_size


def print_time(value, value_type, _print=True):
    """ Convert secondss or milliseconds to the format {d, h:m:s}.
    @param: value_type indicates whether it is seconds or mseconds.
    """

    assert (value_type == "ms" or value_type == "s"), "Value type should be either 'ms' or 's'"

    if value_type == "ms":
        value /= 1000

    s = (value) % 60
    m = (int)((value) / 60) % 60
    h = (int)(((value) / 60) / 60) % 24
    d = (int)(((value) / 60) / 60) / 24

    if _print:
        print("Converting {} {} to human readable form."
              .format(value, "miliseconds" if value_type == "ms" else "seconds"))
        print("{}d {}h:{}m:{}s".format(d, h, m, s))

    return (s, m, h, d)


def persistence_forecast(dataset):
    """ This is a "naive" forecasting method that serves as a baseline for comparisons with the LSTM.
    It effectively returns the last observation t-1 as the forecasted value for timestep t.
    """

    predictions = []
    for i in range(dataset.shape[0]):
        predictions.append(dataset[i, -1, 0])

    return np.array(predictions)


def iterative_prediction(model, last_input, timesteps, steps_ahead):
    """ Predicts h-steps ahead by using an iterative approach. """

    # placeholder for the predictions
    predictions = []
    # reshape window to have the appropriate dims for the LSTM/GRU
    window = last_input.reshape(1, last_input.shape[0], last_input.shape[1])

    for i in range(steps_ahead):
        pred = model.predict(window)
        # print("window: {}, prediction: {}".format(window, pred))
        # shift the window of values and append the prediction
        window = np.roll(window, -1, axis=1)
        window[0][timesteps - 1] = pred[0]
        predictions.append(pred[0])

    return predictions


def plot_losses(metrics):
    """ Plots the training_loss and the validation loss in the same figure. """

    tex_name = "train_valid_loss.tex"

    train_loss = metrics.get("loss")
    valid_loss = metrics.get("val_loss")

    plt.subplot(1, 1, 1)
    plt.plot(train_loss, color='black', label='Train Loss')
    plt.plot(valid_loss, color='red', label='Validation Loss')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Training - Validation Loss for each Epoch")
    plt.legend()

    convert_plot_to_tex(tex_name)

    plt.show()


def plot_predictions(train_true, train_pred, test_true, test_pred, plot=2):
    """ Plots the actual values vs the predictions.
    - @plot: Integer determing which data set to plot. (0=train, 1=test, 2=both)
    """

    train_tex = "train_preds.tex"
    test_tex = "test_preds.tex"

    if plot == 2:
        plt.subplot(1, 1, 1)
        plt.plot(train_true, color='blue', label='true')
        plt.plot(train_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Train dataset")
        plt.legend()

        convert_plot_to_tex(train_tex)

        plt.show()

        plt.subplot(1, 1, 1)
        plt.plot(test_true, color='blue', label='true')
        plt.plot(test_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Test dataset")
        plt.legend()

        convert_plot_to_tex(test_tex)

        plt.show()

    elif plot == 0:
        plt.subplot(1, 1, 1)
        plt.plot(train_true, color='blue', label='true')
        plt.plot(train_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Train dataset")
        plt.legend()

        convert_plot_to_tex(train_tex)

        plt.show()

    elif plot == 1:
        plt.subplot(1, 1, 1)
        plt.plot(test_true, color='blue', label='true')
        plt.plot(test_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Test dataset")
        plt.legend()

        convert_plot_to_tex(test_tex)

        plt.show()


def set_model_name(checkpoint_name):
    """ Sets a global variable denoting the current model. """

    global model_name

    model_name = checkpoint_name.split(".h5")[0]


def convert_plot_to_tex(tex_name, figureheight=r'0.8\textwidth', figurewidth=r'1.0\textwidth'):
    """ Converts the plot figure into a .tex file. """

    path = "/".join([plot_latex_dir, "-".join([model_name, tex_name])])
    tikz_save(path, figureheight=figureheight, figurewidth=figurewidth)

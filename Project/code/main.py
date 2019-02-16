from __future__ import print_function
from __future__ import division

import model as m
import pgf_pdf as pgf
import utils
import time
import numpy as np

# Set fixed seed in order to get reproducible results
np.random.seed(669)

# Use ggplot style-sheet
# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')


def set_checkpoint_name(options):
    """ Sets the checkpoint name for the h5. """

    name = options.get("checkpoint_name") + "-"
    name += "stateful-" if options.get('stateful') is True else ""
    name += "b" + str(options.get('batch_size')) + "-t" + str(options.get('timesteps')) + "-"
    name += "e" + str(options.get('epochs')) + "-" + options.get("network_type") + "-"
    name += "n" + str(options.get("neurons")) + "-d" + str(options.get("dropout")) + "-"
    name += options.get("optimizer") + "-h" + str(options.get("horizon"))
    name += "-seq2seq" if options.get("seq2seq") is True else ""
    name += ".h5"

    return name


def main():

    # plots = False
    plots = True

    options = {
        # "file": "beijing.csv",
        # "checkpoint_name": "beijing",
        "checkpoint_name": "zurich",
        "file": "zurich-monthly-sunspot-numbers.csv",
        # "checkpoint_name": "zurich",
        # "file": "internet-traffic-data-in-bits-   fr.csv",
        # "checkpoint_name": "internet",
        # "file": "internet-traffic-big.csv",
        # "checkpoint_name": "internet-big",
        # "file": "internet-traffic-big.csv",
        # "checkpoint_name": "internet-big",
        "stateful": True,
        "neurons": [128, 256],
        "dropout": 0.2,
        "timesteps": 24,
        "batch_size": 128,
        "epochs": 30,
        "horizon": 1,
        "seq2seq": False,
        # "optimizer": "adam",
        "optimizer": "rmsprop",
        # "network_type": "LSTM",
        "network_type": "GRU",
        "test_ratio": 0.2,
        "val_ratio": 0.1,
    }
    options["checkpoint_name"] = set_checkpoint_name(options)

    utils.set_model_name(options.get("checkpoint_name"))

    data = utils.load_dataset(options.get("file"), plot=False)

    scaler, train, validation, test = utils.preprocess_data(data,
                                                            options.get("timesteps"),
                                                            options.get("horizon"),
                                                            stateful=options.get("stateful"),
                                                            batch_size=options.get('batch_size'),
                                                            test_ratio=options.get('test_ratio'),
                                                            val_ratio=options.get("val_ratio"))

    print("Train: {}, Validation: {}, Test: {},".format(len(train), len(validation), len(test)))
    print("Batch_size is: ", options.get("batch_size"))

    horizon = options.get("horizon")
    seq2seq = options.get("seq2seq")

    trainX, trainY = train.values[:, :-horizon], train.values[:, -horizon:]
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)

    validationX, validationY = validation.values[:, :-horizon], validation.values[:, -horizon:]
    validationX = validationX.reshape(validationX.shape[0], validationX.shape[1], 1)

    testX, testY = test.values[:, :-horizon], test.values[:, -horizon:]
    testX = testX.reshape(testX.shape[0], testX.shape[1], 1)

    if seq2seq:
        print("Because the model is many-to-many we convert Y targets to 3d...")
        trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], 1)
        validationY = validationY.reshape(validationY.shape[0], validationY.shape[1], 1)
        testY = testY.reshape(testY.shape[0], testY.shape[1], 1)

    print("TrainX:", trainX.shape)
    print("TrainY:", trainY.shape)
    print("ValidX:", validationX.shape)
    print("ValidY:", validationY.shape)
    print("TestX: ", testX.shape)
    print("TestY: ", testY.shape)

    model_loaded = False

    if not m.model_exists(options.get("checkpoint_name")):
        print("Model doesn't exist...")

        model = m.build_model(options.get("network_type"), options.get("neurons"), options.get("dropout"),
                              options.get("optimizer"), options.get("batch_size"), options.get("timesteps"),
                              options.get("horizon"), options.get("checkpoint_name"), stateful=options.get("stateful"),
                              seq2seq=options.get("seq2seq"), plot=True)

        callbacks = m.get_callbacks(options.get("checkpoint_name"), options.get("stateful"))

        metrics = {"loss": [], "val_loss": []}

        print("Starting to fit training data to the model...")
        start = time.time()

        history = model.fit(trainX,
                            trainY,
                            validation_data=(validationX, validationY),
                            batch_size=options.get("batch_size"),
                            epochs=options.get("epochs"),
                            verbose=1,
                            shuffle=False,
                            callbacks=callbacks)

        metrics["loss"] = history.history.get("loss")
        metrics["val_loss"] = history.history.get("val_loss")

        end = time.time()
        print("Model fit finished in: ")
        utils.print_time(end - start, "s")

        # Convert all metrics to 4 decimal places
        for key in metrics.keys():
            metrics[key] = map(lambda x: round(x, 4), metrics[key])

    else:
        print("Loading trained model...")
        model = m.get_model(options.get("checkpoint_name"))
        try:
            metrics = m.read_log(options.get("checkpoint_name"))
        except IOError:
            print("Couldn't load metrics. The file (.csv) doesn't exist.")
            metrics = None
        model_loaded = True

    # After training load the best model from the h5 file
    if not model_loaded:
        print("Loading best model...")
        model = m.get_model(options.get("checkpoint_name"))

    # Make predictions
    train_predictions = model.predict(trainX, batch_size=options.get("batch_size"))
    test_predictions = model.predict(testX, batch_size=options.get("batch_size"))

    if seq2seq:
        # In case seq2seq need to reshape the predictions as well as the targets to 2d for sklearn
        train_predictions = train_predictions.reshape(train_predictions.shape[0], train_predictions.shape[1])
        test_predictions = test_predictions.reshape(test_predictions.shape[0], test_predictions.shape[1])
        trainY = trainY.reshape(trainY.shape[0], trainY.shape[1])
        testY = testY.reshape(testY.shape[0], testY.shape[1])

    train_predictions = utils.invert_scale_series(scaler, train_predictions)
    test_predictions = utils.invert_scale_series(scaler, test_predictions)
    trainY = utils.invert_scale_series(scaler, trainY.reshape(-1, horizon))
    testY = utils.invert_scale_series(scaler, testY.reshape(-1, horizon))

    utils.print_statistic_metrics(trainY, train_predictions, testY, test_predictions)

    testX = utils.invert_scale_series(scaler, testX.reshape(-1, horizon))

    # print("Prediction - Actual")
    # for i in range(5):
    #     print("{} - {}".format(test_predictions[i], trainY[i]))
    #     print("{}".format(testX[i:i + 24]))

    if plots:
        if metrics is not None:
            # utils.plot_losses(metrics)
            pgf.plot_losses(metrics)
        # utils.plot_predictions(trainY, train_predictions, testY, test_predictions)
        pgf.plot_predictions(trainY[:200], train_predictions[:200], testY[:200], test_predictions[:200], num_points=200)
        pgf.plot_predictions(trainY[:100], train_predictions[:100], testY[:100], test_predictions[:100], num_points=100)
        pgf.plot_predictions(trainY[:300], train_predictions[:300], testY[:300], test_predictions[:300], num_points=300)
        pgf.plot_predictions(trainY[:500], train_predictions[:500], testY[:500], test_predictions[:500], num_points=500)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        import sys
        sys.exit(1)

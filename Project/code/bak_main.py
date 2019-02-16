from __future__ import print_function
from __future__ import division

import model as m
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

    options = {
        # "file": "beijing.csv",
        # "checkpoint_name": "zurich",
        # "file": "zurich-monthly-sunspot-numbers.csv",
        # "checkpoint_name": "zurich",
        "file": "internet-traffic-data-in-bits-fr.csv",
        "checkpoint_name": "internet",
        "stateful": True,
        "neurons": [128, 256],
        "dropout": 0.2,
        "timesteps": 24,
        "batch_size": 32,
        "epochs": 2,
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

        utils.plot_losses(metrics)
    else:
        print("Loading trained model...")
        model = m.get_model(options.get("checkpoint_name"))
        model_loaded = True

    # After training load the best model from the h5 file
    if not model_loaded:
        print("Loading best model...")
        model = m.get_model(options.get("checkpoint_name"))

    # Make predictions
    train_predictions = model.predict(trainX, batch_size=options.get("batch_size"))
    test_predictions = model.predict(testX, batch_size=options.get("batch_size"))

    # Persistence forecast
    # test_persistence_predictions = utils.persistence_forecast(testX)
    # test_persistence_predictions = utils.invert_scale_series(scaler, test_persistence_predictions.reshape(-1, horizon))

    # Effectively applies the iterative approach in order to forecast h-steps ahead with 1-step forecasting model
    # TODO In order for a stateful to do that we need to define new model and share weights
    # steps_ahead = 10
    # print("-------------------- -------------------- --------------------")
    # print("Predicting {} steaps ahead in the future....".format(steps_ahead))
    # h_preds = utils.iterative_prediction(model, testX[0], options.get('timesteps'), steps_ahead)
    # for index, value in enumerate(h_preds):
    #     # print("step: {}, prediction: {}".format(index, value))
    #     print("step: {}, iter_prediction: {}, normal_prediction: {}".format(index, value, test_predictions[index]))
    # print("-------------------- -------------------- --------------------")

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
    utils.plot_predictions(trainY, train_predictions, testY, test_predictions)

    # utils.print_statistic_metrics(trainY, train_predictions, testY, test_persistence_predictions)

    # print("True - Good - Persistence")
    # for i in range(testY.shape[0]):
    #     print("{} - {} - {}".format(testY[i][0], train_predictions[i][0], test_persistence_predictions[i][0]))


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        import sys
        sys.exit(1)

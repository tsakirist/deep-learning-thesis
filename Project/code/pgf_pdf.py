# -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use("pgf")
pgf_with_rc_fonts = {
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts
    # "axes.labelsize": 1,
    # "xtick.labelsize": 4,               # a little smaller
    # "ytick.labelsize": 4
    "figure.figsize": [8, 5],
    "legend.loc": "best",
}
mpl.rcParams.update(pgf_with_rc_fonts)
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import sys

sns.set(color_codes=True)

plot_latex_dir = "./plots_latex"
plots_pdf_dir = "./plots_pdf"


def datasets_to_pdf_pgf():
    """ Plots all the datasets and converts them to png. """

    datasets = [
        "beijing.csv",
        # "internet-traffic-big.csv"
        # "zurich-monthly-sunspot-numbers.csv"
    ]

    for dataset in datasets:
        data = utils.load_dataset(dataset)

        if "beijing" in dataset:
            title = "Beijing PM2.5 Concentration"
            xlabel = "Date index"
            ylabel = "PM2.5 readings"
            name = "beijing.pdf"
        elif "zurich" in dataset:
            title = "Zurich Monthly Sunspot Numbers"
            xlabel = "Date index"
            ylabel = "Monthly sunspots count"
            name = "zurich.pdf"
        elif "internet" in dataset:
            title = "Internet Traffic Data"
            xlabel = "Date index"
            ylabel = "Averaged internet traffic (bits)"
            name = "internet.pdf"
        else:
            raise ValueError("Can't determine the title of the dataset.")

        fig = plt.figure()
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        # Rotate date to look better
        fig.autofmt_xdate()

        final_path = "./" + name
        print(final_path)
        plt.savefig(final_path)
        plt.close()


def plot_predictions(train_true, train_pred, test_true, test_pred, plot=2, num_points=None):
    """ Makes predictions vs actuals to pgf. """

    train_suffix = "train_preds.pdf"
    test_suffix = "test_preds.pdf"

    if num_points is not None:
        train_suffix = str(num_points) + "_" + train_suffix
        test_suffix = str(num_points) + "_" + test_suffix

    train_final_path = "/".join([plots_pdf_dir, train_suffix])
    test_final_path = "/".join([plots_pdf_dir, test_suffix])

    if plot == 2:
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(train_true, color='blue', label='true')
        plt.plot(train_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Train dataset")
        plt.legend()

        plt.savefig(train_final_path)
        plt.close()

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(test_true, color='blue', label='true')
        plt.plot(test_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Test dataset")
        plt.legend()

        plt.savefig(test_final_path)
        plt.close()

    elif plot == 0:
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(train_true, color='blue', label='true')
        plt.plot(train_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Train dataset")
        plt.legend()

        plt.savefig(train_final_path)
        plt.close()

    elif plot == 1:
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.plot(test_true, color='blue', label='true')
        plt.plot(test_pred, color='red', label='predictions')
        plt.title("Actuals vs Predictions, Test dataset")
        plt.legend()

        plt.savefig(test_final_path)
        plt.close()


def plot_losses(metrics):
    """ Plots the training and validation loss to .pdf-.pgf. """

    name = "train_valid_loss.pdf"
    final_path = "/".join([plots_pdf_dir, name])

    train_loss, valid_loss = metrics.get("loss"), metrics.get("val_loss")

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(train_loss, color='black', label='Training loss')
    plt.plot(valid_loss, color='red', label='Validation loss')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Training - Validation loss for each epoch")
    plt.legend()

    plt.savefig(final_path)
    plt.close()


def main():
    """ Driver program. """
    datasets_to_pdf_pgf()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

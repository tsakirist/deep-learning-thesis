from __future__ import print_function
from __future__ import division

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save
import utils
import sys

plot_latex_dir = "./plots_latex"
datasets_figures_dir = "./figures/datasets"

# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')
sns.set(color_codes=True)

# Set user defined parameters for pyplot
params = {
    "figure.titlesize": 18,
    "figure.figsize": (14, 9),
    # "savefig.facecolor": "white",
    "axes.labelsize": 14,
    # "axes.facecolor": "white",
    "lines.linewidth": 1.5,
    "legend.shadow": True,
    "legend.loc": "best",
}
plt.rcParams.update(params)


def datasets_to_png():
    """ Plots all the datasets and converts them to png. """

    datasets = [
        "beijing.csv",
        "zurich-monthly-sunspot-numbers.csv",
        "internet-traffic-data-in-bits-fr.csv"
    ]

    for dataset in datasets:
        data = utils.load_dataset(dataset)

        if "beijing" in dataset:
            title = "Beijing PM2.5 Concentration"
            xlabel = "Date index"
            ylabel = "PM2.5 readings"
            name = "beijing.png"
        elif "zurich" in dataset:
            title = "Zurich Monthly Sunspot Numbers"
            xlabel = "Date index"
            ylabel = "Monthly sunspots count"
            name = "zurich.png"
        elif "internet" in dataset:
            title = "Internet Traffic Data"
            xlabel = "Date index"
            ylabel = "Averaged internet traffic (bits)"
            name = "internet.png"
        else:
            raise ValueError("Can't determine the title of the dataset.")

        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        final_path = "/".join([datasets_figures_dir, name])
        plt.savefig(final_path)
        plt.close()


def datasets_to_tex(figureheight=r'0.8\textwidth', figurewidth=r'1.0\textwidth'):
    """ Plots all the datasets and converts them to tex. """

    datasets = [
        "beijing.csv",
        "zurich-monthly-sunspot-numbers.csv",
        "internet-traffic-data-in-bits-fr.csv"
    ]

    path = "/".join([plot_latex_dir, "datasets"])

    for dataset in datasets:
        data = utils.load_dataset(dataset)

        if "beijing" in dataset:
            title = "Beijing PM2.5 Concentration"
            tex_name = "beijing_plot.tex"
        elif "zurich" in dataset:
            title = "Zurich Monthly Sunspot Numbers"
            tex_name = "zurich_plot.tex"
        elif "internet" in dataset:
            title = "Internet Traffic Data"
            tex_name = "internet_plot.tex"
        else:
            raise ValueError("Can't determine the title of the dataset.")

        plt.plot(data)
        plt.title(title)

        final_path = "/".join([path, tex_name])
        tikz_save(final_path, figureheight=figureheight, figurewidth=figurewidth)

        print("Converted {} to .tex format".format(title))
        print("Saving it at {}".format(final_path))


def main():
    """ Driver program. """

    if len(sys.argv) > 2:
        raise ValueError("Too many arguments. Provide only one.")
    elif len(sys.argv) == 1:
        raise ValueError("Choose whether .png or .tex output for the datasets.")

    if sys.argv[1] == "png":
        print("----------- Saving plotted datasets to .png -----------")
        datasets_to_png()
    elif sys.argv[1] == "tex":
        print("----------- Saving plotted datasets to .tex -----------")
        datasets_to_tex()
    else:
        raise ValueError("Wrong argument. Available options (png, tex).")


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)

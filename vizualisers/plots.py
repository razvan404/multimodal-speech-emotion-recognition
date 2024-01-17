from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class PlotVisualizer:
    @classmethod
    def plot_history(cls, history: list[float], title: str):
        plt.plot(history)
        plt.title(title)

    @classmethod
    def plot_many(cls, dims: (int, int), *args):
        assert dims[0] * dims[1] == len(args)
        for idx, arg in enumerate(args, start=1):
            plt.subplot(*dims, idx)
            arg()
        plt.show()

    @classmethod
    def plot_confusion_matrix(
        cls, confusion_matrix: list[list[int]], labels: list[str]
    ):
        ConfusionMatrixDisplay(confusion_matrix, display_labels=labels).plot()
        plt.show()

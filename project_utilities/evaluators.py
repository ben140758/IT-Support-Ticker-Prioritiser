from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from matplotlib.pyplot import show, subplots, get_current_fig_manager
from pandas import DataFrame
from numpy import sum as numpy_sum, ndarray, empty_like


class ITSupportPriorityConfusionMatrixEvaluator:
    """Class for storing and showing a confusion matrix.

    Adapted from https://www.kaggle.com/code/agungor2/various-confusion-matrix-plots/notebook"""
    dataset_confusion_matrix = confusion_matrix
    dataset_confusion_matrix_sums = confusion_matrix
    dataset_confusion_matrix_percentages = confusion_matrix
    dataset_annotations = ndarray
    predictions = tuple or ndarray
    actual_values = tuple or ndarray
    labels = list

    def __init__(self, predictions: tuple or ndarray, actual_values: tuple or ndarray, labels: list):
        self.predictions = predictions
        self.actual_values = actual_values
        self.labels = labels
        print(self.labels)
        self.dataset_confusion_matrix = confusion_matrix(self.actual_values, self.predictions, labels=self.labels)
        self.dataset_annotations = empty_like(self.dataset_confusion_matrix).astype(str)
        self.confusion_matrix_sums = numpy_sum(self.dataset_confusion_matrix, axis=1, keepdims=True)
        self.confusion_matrix_percentages = self.dataset_confusion_matrix / self.confusion_matrix_sums.astype(
            float) * 100

    def plot_confusion_matrix(self, fullscreen_requested: bool = False):
        self.__update_dataset_annotations()
        dataset_confusion_matrix_data_frame = DataFrame(self.dataset_confusion_matrix,
                                                        index=self.labels,
                                                        columns=self.labels)
        dataset_confusion_matrix_data_frame.index.name = 'Actual'
        dataset_confusion_matrix_data_frame.columns.name = 'Predicted'
        label_quantity = len(self.labels)
        fig, ax = subplots(figsize=(label_quantity, label_quantity))

        # Adapted from https://stackoverflow.com/questions/42111075/seaborn-heatmap-color-scheme-based-on-row-values
        normalised_confusion_matrix = dataset_confusion_matrix_data_frame.div(
            dataset_confusion_matrix_data_frame.max(axis=1), axis=0)
        heatmap(normalised_confusion_matrix, cmap="YlGnBu", annot=self.dataset_annotations, fmt='', ax=ax)

        # Adapted from https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
        # (dinvlad)
        if fullscreen_requested:
            fig_manager = get_current_fig_manager()
            fig_manager.window.state('zoomed')

        show()

    def __update_dataset_annotations(self):
        n_rows, n_columns = self.dataset_confusion_matrix.shape
        [self.alter_annotation(row, column) for row in range(n_rows) for column in range(n_columns)]

    def alter_annotation(self, row: int, column: int):
        cell_predicted_count = self.dataset_confusion_matrix[row, column]
        cell_percentage_of_category = self.confusion_matrix_percentages[row, column]
        category_count = self.confusion_matrix_sums[row]

        if row == column or cell_predicted_count != 0:
            self.dataset_annotations[row, column] = '%.1f%%\n%d/%d' % (
                cell_percentage_of_category, cell_predicted_count, category_count)
        else:
            self.dataset_annotations[row, column] = '%d%%\n%d/%d' % (0, 0, category_count)


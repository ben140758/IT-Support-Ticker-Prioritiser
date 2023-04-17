from sklearn.metrics import confusion_matrix
from seaborn import heatmap
import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import sum as numpy_sum, ndarray, empty_like
from dataclasses import dataclass

global CURRENT_FIGURES


class DetailedConfusionMatrix:
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
        fig, ax = plt.subplots(figsize=(label_quantity, label_quantity))

        # Adapted from https://stackoverflow.com/questions/42111075/seaborn-heatmap-color-scheme-based-on-row-values
        normalised_confusion_matrix = dataset_confusion_matrix_data_frame.div(
            dataset_confusion_matrix_data_frame.max(axis=1), axis=0)
        heatmap(normalised_confusion_matrix, cmap="YlGnBu", annot=self.dataset_annotations, fmt='', ax=ax)

        # Adapted from https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
        # (dinvlad)
        if fullscreen_requested:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.state('zoomed')

        plt.show()

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
@dataclass
class AccuracyPerClass:
    label_predictions: list
    actual_labels: list
    label_classes: list

    @property
    def confusion_matrix(self):
        # Get the confusion matrix
        cm = confusion_matrix(self.actual_labels, self.label_predictions)
        return cm

    def sort_to_correct_incorrect_predictions(self):
        correct_predictions = {'P5': 0, 'P4': 0, 'P3': 0, 'P2': 0, 'P1': 0}
        incorrect_predictions = {'P5': 0, 'P4': 0, 'P3': 0, 'P2': 0, 'P1': 0}
        for predicted, actual in zip(self.label_predictions, self.actual_labels):
            if predicted == actual:
                correct_predictions[actual] += 1
            else:
                incorrect_predictions[actual] += 1

        return list(correct_predictions.values()), list(incorrect_predictions.values())

    def normalise_correct_incorrect(self, correct, incorrect):
        total_predictions_per_label = [correct[label] + incorrect[label] for label in range(len(correct))]
        normalised_correct_predictions = [correct[label] / total_predictions_per_label[label] for label in range(len(correct))]
        normalised_incorrect_predictions = [incorrect[label] / total_predictions_per_label[label] for label in range(len(incorrect))]
        return normalised_correct_predictions, normalised_incorrect_predictions
    def plot_confusion_matrix(self):
        """
        Adapted from firstly phind prompts:
            1. How do you directly compare two confusion matrices
            2. Generate a python script that shows percentage accuracy of 5 different classes
            3. Can you generate the code to plot this with matplotlib
            4. Make the script plot the percentage accuracy of each class

        Then Bing AI GPT Prompts:
            1. generate a python function that plots correct and incorrect for a specified number of classes
            2. could you normalize each class so the bars are equal

        :return: None
        """
        # Create bar plot
        labels = [f'{self.label_classes[i]}' for i in range(len(self.label_classes))]
        correct, incorrect = self.sort_to_correct_incorrect_predictions()
        normalised_correct, normalised_incorrect = self.normalise_correct_incorrect(correct, incorrect)
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(labels, normalised_correct, width, label='Correct')
        ax.bar(labels, normalised_incorrect, width, bottom=normalised_correct, label='Incorrect')
        ax.set_ylabel('Correct - Incorrect Proportion')
        ax.legend()
        plt.show()
